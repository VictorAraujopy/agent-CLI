"""Tests for the generation with a local model."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from agent_cli.inference.generation import LocalModel
from agent_cli.inference.loader import load_model, open_without_touching_the_terminal
from agent_cli.inference.model import ContextFullError

STORY_START = [{"role": "user", "content": "Once upon a time"}]
PIECES_TO_READ = 5
TOKENS_LEFT_FREE = 64


@pytest.fixture(scope="module")
def local_model(small_model_file: Path) -> LocalModel:
    """Return the small model, loaded once for every test of this file."""
    return LocalModel(load_model(small_model_file))


def chunk_with(delta: dict[str, str], finish_reason: str | None = None) -> dict[str, Any]:
    """Return one streamed chunk in the shape llama.cpp gives, carrying the given delta and finish reason."""
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def conversation_too_long_for(model: LocalModel) -> list[dict[str, str]]:
    """Return a conversation with more words than the model's context has tokens, so it can never fit."""
    return [{"role": "user", "content": "Once upon a time " * model.context_limit()}]


def conversation_with_little_room_left(model: LocalModel) -> list[dict[str, str]]:
    """Return a conversation that fills the model's context but for a few dozen tokens."""
    sample = "Once upon a time " * 100
    tokens_per_repeat = len(model.llama.tokenize(sample.encode(), add_bos=False)) / 100
    repeats = int((model.context_limit() - TOKENS_LEFT_FREE) / tokens_per_repeat)
    return [{"role": "user", "content": "Once upon a time " * repeats}]


class FailingStream:
    """A stream that gives some chunks and then fails, the way the library fails while it reads or writes."""

    def __init__(self, error: Exception, chunks_before_failing: list[dict[str, Any]] | None = None) -> None:
        """Keep the chunks to give first, none by default, and the error to raise after them."""
        self.error = error
        self.chunks_before_failing = chunks_before_failing or []

    def __iter__(self) -> "FailingStream":
        """Return the stream itself, which gives the chunks."""
        return self

    def __next__(self) -> dict[str, Any]:
        """Give the next chunk written for it, or raise the error once there are none left."""
        if self.chunks_before_failing:
            return self.chunks_before_failing.pop(0)
        raise self.error


def test_reply_arrives_in_pieces_of_text(local_model: LocalModel) -> None:
    """The reply comes one piece at a time, each one a text that is not empty, and can be closed."""
    reply = local_model.reply(STORY_START)
    first_pieces = list(islice(reply, PIECES_TO_READ))
    reply.close()
    assert len(first_pieces) == PIECES_TO_READ
    assert all(isinstance(piece, str) and piece for piece in first_pieces)


def test_model_generates_again_after_a_reply_is_closed(local_model: LocalModel) -> None:
    """Closing a reply in the middle leaves the model ready for the next one."""
    closed_reply = local_model.reply(STORY_START)
    next(closed_reply)
    closed_reply.close()
    next_reply = local_model.reply(STORY_START)
    assert next(next_reply)
    next_reply.close()


def test_chunks_without_text_give_no_piece(local_model: LocalModel, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(local_model.llama, "create_chat_completion", stream_the_chunks)
    assert list(local_model.reply(STORY_START)) == ["Hel", "lo"]


def test_reply_is_asked_as_a_stream_with_no_length_limit(
    local_model: LocalModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library is asked to stream, and the reply gets no limit, so it ends when the model stops."""
    options_given_to_llama_cpp = {}

    def remember_the_options(messages: list[dict[str, str]], **options: Any) -> Iterator[dict[str, Any]]:
        """Stand in for the library: record the options it was given and stream nothing."""
        options_given_to_llama_cpp.update(options)
        return iter([])

    monkeypatch.setattr(local_model.llama, "create_chat_completion", remember_the_options)
    list(local_model.reply(STORY_START))
    assert options_given_to_llama_cpp["stream"] is True
    assert "max_tokens" not in options_given_to_llama_cpp


def test_whole_conversation_reaches_the_library(local_model: LocalModel, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(local_model.llama, "create_chat_completion", remember_the_messages)
    list(local_model.reply(conversation))
    assert messages_given_to_llama_cpp == conversation


def test_conversation_longer_than_the_context_is_a_context_full_error(local_model: LocalModel) -> None:
    """A conversation the context cannot hold stops with ContextFullError, before any piece of reply."""
    reply = local_model.reply(conversation_too_long_for(local_model))
    with pytest.raises(ContextFullError):
        next(reply)


def test_other_errors_of_the_library_are_not_taken_for_a_full_context(
    local_model: LocalModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError of the library that is not about the context passes through as it came."""

    def stream_that_fails(messages: list[dict[str, str]], **options: Any) -> FailingStream:
        """Stand in for the library: a stream that fails with an error unrelated to the context."""
        return FailingStream(ValueError("the grammar is not valid"))

    monkeypatch.setattr(local_model.llama, "create_chat_completion", stream_that_fails)
    with pytest.raises(ValueError, match="the grammar is not valid"):
        next(local_model.reply(STORY_START))


def test_model_replies_again_after_a_conversation_that_did_not_fit(local_model: LocalModel) -> None:
    """After refusing a conversation too long for the context, the model replies to one that fits."""
    with pytest.raises(ContextFullError):
        next(local_model.reply(conversation_too_long_for(local_model)))
    reply = local_model.reply(STORY_START)
    assert next(reply)
    reply.close()


def test_stop_ends_the_reply_without_an_error(local_model: LocalModel) -> None:
    """After stop(), llama.cpp stops at its next batch and the reply simply ends, with no error."""
    reply = local_model.reply(STORY_START)
    next(reply)
    local_model.stop()
    assert list(reply) == []


def test_model_replies_again_after_a_stop(local_model: LocalModel) -> None:
    """A stopped reply leaves the model ready for the next one."""
    stopped_reply = local_model.reply(STORY_START)
    next(stopped_reply)
    local_model.stop()
    list(stopped_reply)
    next_reply = local_model.reply(STORY_START)
    assert next(next_reply)
    next_reply.close()


def test_stop_asked_between_replies_does_not_end_the_next_one(local_model: LocalModel) -> None:
    """A stop asked while no reply is being written is forgotten when the next reply starts."""
    local_model.stop()
    reply = local_model.reply(STORY_START)
    assert next(reply)
    reply.close()


def test_other_error_while_a_stop_is_pending_passes_through(
    local_model: LocalModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure of llama.cpp that is not the stop is raised, even when a stop was asked."""

    def stream_that_fails(messages: list[dict[str, str]], **options: Any) -> FailingStream:
        """Stand in for the library: give one piece, then fail with an error that is not the stop."""
        return FailingStream(RuntimeError("llama_decode returned -3"), [chunk_with({"content": "Hel"})])

    monkeypatch.setattr(local_model.llama, "create_chat_completion", stream_that_fails)
    reply = local_model.reply(STORY_START)
    next(reply)
    local_model.stop()
    with pytest.raises(RuntimeError, match="returned -3"):
        next(reply)


def test_stop_nobody_asked_for_passes_through(local_model: LocalModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """llama.cpp saying it stopped, when stop() was not asked, is raised as the failure it is."""

    def stream_that_fails(messages: list[dict[str, str]], **options: Any) -> FailingStream:
        """Stand in for the library: fail saying the work was stopped."""
        return FailingStream(RuntimeError("llama_decode returned 2"))

    monkeypatch.setattr(local_model.llama, "create_chat_completion", stream_that_fails)
    with pytest.raises(RuntimeError, match="returned 2"):
        next(local_model.reply(STORY_START))


def test_reply_that_runs_out_of_context_is_cut(local_model: LocalModel) -> None:
    """A reply that ends because the context filled is reported as cut. The small model never ends a story by itself."""
    list(local_model.reply(conversation_with_little_room_left(local_model)))
    assert local_model.reply_was_cut()


def test_reply_the_model_finishes_is_not_cut(local_model: LocalModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply whose last chunk says the model stopped is not reported as cut."""
    chunks = [chunk_with({"content": "The end."}), chunk_with({}, finish_reason="stop")]

    def stream_the_chunks(messages: list[dict[str, str]], **options: Any) -> Iterator[dict[str, Any]]:
        """Stand in for the library: stream the chunks written above."""
        return iter(chunks)

    monkeypatch.setattr(local_model.llama, "create_chat_completion", stream_the_chunks)
    list(local_model.reply(STORY_START))
    assert not local_model.reply_was_cut()


def test_closed_reply_is_not_cut_even_after_a_cut_one(local_model: LocalModel) -> None:
    """A reply closed in the middle did not run out of context, whatever happened to the reply before it."""
    list(local_model.reply(conversation_with_little_room_left(local_model)))
    closed_reply = local_model.reply(STORY_START)
    next(closed_reply)
    closed_reply.close()
    assert not local_model.reply_was_cut()


def test_context_holds_the_whole_prompt_with_what_the_chat_format_adds(local_model: LocalModel) -> None:
    """Once the reply starts, the count is the prompt the library sent, not just the text of the messages."""
    prompt_sent = local_model.llama.create_chat_completion(STORY_START, max_tokens=1)["usage"]["prompt_tokens"]
    reply = local_model.reply(STORY_START)
    next(reply)
    reply.close()
    assert local_model.tokens_in_use() == prompt_sent


def test_context_grows_with_every_piece_of_the_reply(local_model: LocalModel) -> None:
    """While the reply is written, each new piece adds one token to the context."""
    reply = local_model.reply(STORY_START)
    list(islice(reply, PIECES_TO_READ))
    count_before = local_model.tokens_in_use()
    list(islice(reply, PIECES_TO_READ))
    count_after = local_model.tokens_in_use()
    reply.close()
    assert count_after - count_before == PIECES_TO_READ


def test_context_limit_is_the_context_stored_in_the_gguf(local_model: LocalModel) -> None:
    """The limit is the context size written in the model's .gguf, the one it was trained with."""
    architecture = local_model.llama.metadata["general.architecture"]
    trained_context = int(local_model.llama.metadata[f"{architecture}.context_length"])
    assert local_model.context_limit() == trained_context


def test_context_limit_never_passes_the_context_the_model_was_trained_with(small_model_file: Path) -> None:
    """llama.cpp rounds the context it creates up to a step of 256, but the limit stays at the trained size."""
    trained_context = 1000
    llama = open_without_touching_the_terminal(
        model_path=str(small_model_file),
        n_ctx=0,
        kv_overrides={"llama.context_length": trained_context},
    )
    assert llama.n_ctx() == 1024
    assert LocalModel(llama).context_limit() == trained_context
