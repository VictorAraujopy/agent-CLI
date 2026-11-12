"""Generation with a local model: the pieces of the reply, and how many tokens a conversation takes."""

from collections.abc import Iterator
from typing import cast

import llama_cpp
from llama_cpp import CreateChatCompletionStreamResponse, Llama


def generate_reply(model: Llama, messages: list[dict[str, str]]) -> Iterator[str]:
    """Yield the reply to the conversation one piece of text at a time, as the model writes it."""
    # Without max_tokens the reply ends when the model stops or the context is full.
    response = model.create_chat_completion(messages, stream=True)
    # The library declares one return type for both modes; with stream=True it is always the chunks.
    stream = cast(Iterator[CreateChatCompletionStreamResponse], response)
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        # The first chunk only says the reply is the assistant's, and the last one only says why the
        # reply ended. Neither carries text.
        text = delta.get("content")
        if text:
            yield text


def count_tokens(model: Llama, messages: list[dict[str, str]]) -> int:
    """Return how many tokens the text of the messages takes, counted with the model's own tokenizer."""
    total = 0
    for message in messages:
        text = message["content"].encode()
        total += len(model.tokenize(text, add_bos=False))
    return total


def context_limit(model: Llama) -> int:
    """Return how many tokens the conversation may take: the context the model was trained with."""
    # llama.cpp creates contexts in steps of 256 tokens, so the one it created can pass the trained size.
    created_context = model.n_ctx()
    trained_context = llama_cpp.llama_model_n_ctx_train(model.model)
    return min(created_context, trained_context)
