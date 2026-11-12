"""Tests for the generation with a local model."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pytest
from llama_cpp import Llama

from agent_cli.inference.generation import context_limit, generate_reply, tokens_in_context
from agent_cli.inference.loader import load_model, open_without_touching_the_terminal

STORY_START = [{"role": "user", "content": "Once upon a time"}]
PIECES_TO_READ = 5


@pytest.fixture(scope="module")
def small_model(small_model_file: Path) -> Llama:
    """Return the small model, loaded once for every test of this file."""
    return load_model(small_model_file)


def chunk_with(delta: dict[str, str]) -> dict[str, Any]:
    """Return one streamed chunk in the shape llama.cpp gives, carrying the given delta."""
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def test_reply_arrives_in_pieces_of_text(small_model: Llama) -> None:
    """The reply comes one piece at a time, each one a text that is not empty, and can be stopped."""
    reply = generate_reply(small_model, STORY_START)
    first_pieces = list(islice(reply, PIECES_TO_READ))
    reply.close()
    assert len(first_pieces) == PIECES_TO_READ
    assert all(isinstance(piece, str) and piece for piece in first_pieces)


def test_model_generates_again_after_a_reply_is_stopped(small_model: Llama) -> None:
    """Stopping a reply in the middle leaves the model ready for the next one."""
    stopped_reply = generate_reply(small_model, STORY_START)
    next(stopped_reply)
    stopped_reply.close()
    next_reply = generate_reply(small_model, STORY_START)
    assert next(next_reply)
    next_reply.close()


def test_chunks_without_text_give_no_piece(small_model: Llama, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunk that only names the role, an empty text and the last chunk become no piece."""
    chunks = [
        chunk_with({"role": "assistant"}),
        chunk_with({"content": "Hel"}),
        chunk_with({"content": ""}),
        chunk_with({"content": "lo"}),
        chunk_with({}),
    ]

    def stream_the_chunks(messages: list[dict[str, str]], **options: Any) -> Iterator[dict[str, Any]]:
        """Stand in for the library: stream the chunks written above."""
        return iter(chunks)

    monkeypatch.setattr(small_model, "create_chat_completion", stream_the_chunks)
    assert list(generate_reply(small_model, STORY_START)) == ["Hel", "lo"]


def test_reply_is_asked_as_a_stream_with_no_length_limit(
    small_model: Llama, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library is asked to stream, and the reply gets no limit, so it ends when the model stops."""
    options_given_to_llama_cpp = {}

    def remember_the_options(messages: list[dict[str, str]], **options: Any) -> Iterator[dict[str, Any]]:
        """Stand in for the library: record the options it was given and stream nothing."""
        options_given_to_llama_cpp.update(options)
        return iter([])

    monkeypatch.setattr(small_model, "create_chat_completion", remember_the_options)
    list(generate_reply(small_model, STORY_START))
    assert options_given_to_llama_cpp["stream"] is True
    assert "max_tokens" not in options_given_to_llama_cpp


def test_whole_conversation_reaches_the_library(small_model: Llama, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every message of the conversation, in order, is handed to the library, not only the last one."""
    conversation = [
        {"role": "system", "content": "You tell stories."},
        {"role": "user", "content": "Once upon a time"},
    ]
    messages_given_to_llama_cpp = []

    def remember_the_messages(messages: list[dict[str, str]], **options: Any) -> Iterator[dict[str, Any]]:
        """Stand in for the library: record the messages it was given and stream nothing."""
        messages_given_to_llama_cpp.extend(messages)
        return iter([])

    monkeypatch.setattr(small_model, "create_chat_completion", remember_the_messages)
    list(generate_reply(small_model, conversation))
    assert messages_given_to_llama_cpp == conversation


def test_context_holds_the_whole_prompt_with_what_the_chat_format_adds(small_model: Llama) -> None:
    """Once the reply starts, the count is the prompt the library sent, not just the text of the messages."""
    prompt_sent = small_model.create_chat_completion(STORY_START, max_tokens=1)["usage"]["prompt_tokens"]
    reply = generate_reply(small_model, STORY_START)
    next(reply)
    reply.close()
    assert tokens_in_context(small_model) == prompt_sent


def test_context_grows_with_every_piece_of_the_reply(small_model: Llama) -> None:
    """While the reply is written, each new piece adds one token to the context."""
    reply = generate_reply(small_model, STORY_START)
    list(islice(reply, PIECES_TO_READ))
    count_before = tokens_in_context(small_model)
    list(islice(reply, PIECES_TO_READ))
    count_after = tokens_in_context(small_model)
    reply.close()
    assert count_after - count_before == PIECES_TO_READ


def test_context_limit_is_the_context_stored_in_the_gguf(small_model: Llama) -> None:
    """The limit is the context size written in the model's .gguf, the one it was trained with."""
    architecture = small_model.metadata["general.architecture"]
    trained_context = int(small_model.metadata[f"{architecture}.context_length"])
    assert context_limit(small_model) == trained_context


def test_context_limit_never_passes_the_context_the_model_was_trained_with(small_model_file: Path) -> None:
    """llama.cpp rounds the context it creates up to a step of 256, but the limit stays at the trained size."""
    trained_context = 1000
    model = open_without_touching_the_terminal(
        model_path=str(small_model_file),
        n_ctx=0,
        kv_overrides={"llama.context_length": trained_context},
    )
    assert model.n_ctx() == 1024
    assert context_limit(model) == trained_context
