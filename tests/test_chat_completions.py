"""Tests for what OpenRouter and llama-server share; the replies themselves are tested through each model."""

from agent_cli.inference.chat_completions import chunks_in


def test_only_data_lines_carry_chunks() -> None:
    """Comments, blank lines and other fields are skipped, and a data line needs no space after the colon."""
    lines = [": comment", "", "event: message", 'data: {"n": 1}', 'data:{"n": 2}', "id: 7"]
    assert list(chunks_in(lines)) == [{"n": 1}, {"n": 2}]


def test_stream_ends_at_the_done_mark() -> None:
    """The mark that ends the stream is not read as a chunk, and nothing after it is read."""
    lines = ['data: {"n": 1}', "data: [DONE]", 'data: {"n": 2}']
    assert list(chunks_in(lines)) == [{"n": 1}]
