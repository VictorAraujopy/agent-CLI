"""Generation with a local model: LocalModel, the Model that runs on this machine through llama.cpp."""

import threading
from collections.abc import Iterator
from typing import Any, cast

import llama_cpp
from llama_cpp import CreateChatCompletionStreamResponse, Llama
from llama_cpp.llama_chat_format import (
    ChatFormatterResponse,
    Jinja2ChatFormatter,
    chat_formatter_to_chat_completion_handler,
)

from agent_cli.inference.model import THINKING_OPENS, ContextFullError, Model

CHAT_TEMPLATE_IN_THE_GGUF = "tokenizer.chat_template"
NO_SUCH_TOKEN = -1


def says_context_is_full(error: ValueError) -> bool:
    """Tell whether an error of the library says the conversation does not fit in the context."""
    # The library raises a plain ValueError for this, so only its text tells it apart from other errors.
    return "exceed context window" in str(error)


def token_text(llama: Llama, token: int) -> str:
    """Return the text of a token of the model, or no text when the model does not have that token."""
    if token == NO_SUCH_TOKEN:
        return ""
    return llama.detokenize([token], special=True).decode()


def formatter_of_the_template(llama: Llama) -> Jinja2ChatFormatter | None:
    """Return what formats a conversation with the chat template stored in the .gguf, or None when it stores none."""
    template = llama.metadata.get(CHAT_TEMPLATE_IN_THE_GGUF)
    if template is None:
        return None
    end_token = llama.token_eos()
    return Jinja2ChatFormatter(
        template=template,
        eos_token=token_text(llama, end_token),
        bos_token=token_text(llama, llama.token_bos()),
        stop_token_ids=[end_token],
    )


class LocalModel(Model):
    """A model loaded from a .gguf file, running here, and on the RPC servers when split, through llama.cpp."""

    def __init__(self, llama: Llama) -> None:
        """Keep the model the loader opened, with no reply written yet, and let llama.cpp ask whether to stop."""
        self.llama = llama
        # Why the last reply ended, as the library says it: "stop" when the model finished it, "length" when
        # the context filled. None while a reply is being written, or when it was stopped before the end.
        self.finish_reason: str | None = None
        self.stop_requested = threading.Event()
        # llama.cpp calls this before each batch it processes, also while it reads a long conversation, and
        # stops there when it answers True. Kept in the model, because Python must not free it while in use.
        self.abort_callback = llama_cpp.ggml_abort_callback(self.should_abort)
        llama_cpp.llama_set_abort_callback(self.llama.ctx, self.abort_callback, None)
        # Whether the template opened the thinking at the end of the last conversation it formatted. The
        # reply then starts inside the thinking, without the opening tag, which is in what the model read.
        self.template_opened_the_thinking = False
        # The library formats the conversation without telling the template to let the model think. With
        # the template of the .gguf, the model formats it here instead. A model with no template keeps
        # the format the library chose for it.
        self.formatter = formatter_of_the_template(llama)
        if self.formatter is not None:
            llama.chat_handler = chat_formatter_to_chat_completion_handler(self.format_with_thinking)

    def format_with_thinking(self, **conversation: Any) -> ChatFormatterResponse:
        """Format the conversation with the model's template, told to let the model think before it answers."""
        # enable_thinking is the name the templates that can turn the thinking off give to the switch;
        # a template that has no such switch ignores it.
        formatted = cast(Jinja2ChatFormatter, self.formatter)(**conversation, enable_thinking=True)
        self.template_opened_the_thinking = formatted.prompt.rstrip().endswith(THINKING_OPENS)
        return formatted

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the reply to the conversation one piece of text at a time, as the model writes it."""
        self.finish_reason = None
        self.stop_requested.clear()
        # Without max_tokens the reply ends when the model stops or the context is full.
        response = self.llama.create_chat_completion(messages, stream=True)
        # The library declares one return type for both modes; with stream=True it is always the chunks.
        stream = cast(Iterator[CreateChatCompletionStreamResponse], response)
        try:
            yield from self.with_the_opening_of_the_thinking(self.text_pieces(stream))
        except ValueError as error:
            if not says_context_is_full(error):
                raise
            # The library measures the conversation before reading it, so no piece was written yet.
            raise ContextFullError(str(error)) from error
        except RuntimeError as error:
            if not self.was_stopped_on_request(error):
                raise
            # Stopped by stop(): the reply ends here, with the pieces written so far.

    def text_pieces(self, stream: Iterator[CreateChatCompletionStreamResponse]) -> Iterator[str]:
        """Yield the text of each streamed chunk, and keep why the reply ended when the last chunk says it."""
        for chunk in stream:
            choice = chunk["choices"][0]
            # The first chunk only says the reply is the assistant's, and the last one only says why the
            # reply ended. Neither carries text.
            if choice["finish_reason"] is not None:
                self.finish_reason = choice["finish_reason"]
            text = choice["delta"].get("content")
            if text:
                yield text

    def with_the_opening_of_the_thinking(self, pieces: Iterator[str]) -> Iterator[str]:
        """Yield the pieces, the opening tag of the thinking first when the template wrote it and not the model."""
        for position, piece in enumerate(pieces):
            if position == 0 and self.template_opened_the_thinking:
                yield f"{THINKING_OPENS}\n"
            yield piece

    def stop(self) -> None:
        """Ask the reply being written to end, with no error, as soon as it can; safe to call from any thread."""
        self.stop_requested.set()

    def should_abort(self, callback_data: int | None) -> bool:
        """Answer llama.cpp, which asks before each batch it processes, whether to stop: yes once stop() asked."""
        return self.stop_requested.is_set()

    def was_stopped_on_request(self, error: RuntimeError) -> bool:
        """Tell whether an error of the library is llama.cpp stopping its work because stop() asked it to."""
        # llama_decode returns 2 when the abort callback said to stop. Both checks matter: another error
        # while a stop is pending, or a 2 nobody asked for, is a real failure.
        return self.stop_requested.is_set() and "llama_decode returned 2" in str(error)

    def reply_was_cut(self) -> bool:
        """Tell whether the last reply ended because the context was full, not because the model finished it."""
        # No reply is given max_tokens, so "length" can only mean the context ran out.
        return self.finish_reason == "length"

    def tokens_in_use(self) -> int:
        """Return how many tokens the context holds now: the last conversation sent and the reply written so far."""
        # The library keeps this count itself, with everything the chat format adds around each message.
        return self.llama.n_tokens

    def context_limit(self) -> int:
        """Return how many tokens the conversation may take: the context created, never above the trained one."""
        # llama.cpp creates contexts in steps of 256 tokens, so the one it created can pass the trained size.
        created_context = self.llama.n_ctx()
        trained_context = llama_cpp.llama_model_n_ctx_train(self.llama.model)
        return min(created_context, trained_context)
