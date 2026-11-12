"""Generation with a local model: LocalModel, the Model that runs on this machine through llama.cpp."""

import threading
from collections.abc import Iterator
from typing import cast

import llama_cpp
from llama_cpp import CreateChatCompletionStreamResponse, Llama

from agent_cli.inference.model import ContextFullError, Model


def says_context_is_full(error: ValueError) -> bool:
    """Tell whether an error of the library says the conversation does not fit in the context."""
    # The library raises a plain ValueError for this, so only its text tells it apart from other errors.
    return "exceed context window" in str(error)


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

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the reply to the conversation one piece of text at a time, as the model writes it."""
        self.finish_reason = None
        self.stop_requested.clear()
        # Without max_tokens the reply ends when the model stops or the context is full.
        response = self.llama.create_chat_completion(messages, stream=True)
        # The library declares one return type for both modes; with stream=True it is always the chunks.
        stream = cast(Iterator[CreateChatCompletionStreamResponse], response)
        try:
            yield from self.text_pieces(stream)
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
        """Return how many tokens the conversation may take: the context the model was trained with."""
        # llama.cpp creates contexts in steps of 256 tokens, so the one it created can pass the trained size.
        created_context = self.llama.n_ctx()
        trained_context = llama_cpp.llama_model_n_ctx_train(self.llama.model)
        return min(created_context, trained_context)
