"""Tests for what every model shares: how a reply is split into thinking and answer."""

import pytest

from agent_cli.inference.model import thinking_and_answer, without_its_thinking


@pytest.mark.parametrize(
    ("reply", "thinking", "answer"),
    [
        ("Hello.", "", "Hello."),
        ("<think>\nA greeting.\n</think>\n\nHello.", "A greeting.", "Hello."),
        ("\n<think>A greeting.</think>Hello.", "A greeting.", "Hello."),
        ("<think>\nstopped while thinking", "stopped while thinking", ""),
        ("The <think> tag opens it and </think> closes it.", "", "The <think> tag opens it and </think> closes it."),
        ("<", "", "<"),
    ],
)
def test_reply_is_split_into_the_thinking_that_starts_it_and_the_answer(reply: str, thinking: str, answer: str) -> None:
    """Thinking is what comes between the tags at the start; a reply stopped inside it has no answer."""
    assert thinking_and_answer(reply) == (thinking, answer)


def test_only_the_replies_of_a_model_lose_their_thinking() -> None:
    """What the user or the system prompt wrote stays as it is, also when it holds the tags."""
    user_message = {"role": "user", "content": "<think>\nnot a thinking\n</think>\n\nSay hello."}
    reply = {"role": "assistant", "content": "<think>\nA greeting.\n</think>\n\nHello."}
    assert without_its_thinking(user_message) == user_message
    assert without_its_thinking(reply) == {"role": "assistant", "content": "Hello."}
