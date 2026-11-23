"""Tests for the queries of the history: turns saved as they end, and conversations read back to continue them."""

import json
from pathlib import Path

import pytest

from agent_cli.history import queries
from agent_cli.history.database import transaction
from agent_cli.history.queries import History, SavedTurn
from agent_cli.tools.tool import Ending, ToolResult

READ_CALL = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}}


def turn_with_a_tool() -> list[dict[str, object]]:
    """Return the messages of a turn whose reply read a file before answering."""
    return [
        {"role": "user", "content": "Read a"},
        {"role": "assistant", "content": "<think>\nplan\n</think>\n\nLooking.", "tool_calls": [READ_CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "hello\n"},
        {"role": "assistant", "content": "It says hello."},
    ]


def exchange(question: str, answer: str) -> list[dict[str, object]]:
    """Return the messages of a turn with no tool."""
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def history_of(tmp_path: Path, folder_name: str = "project") -> History:
    """Return the history of a working folder, kept in a file shared by every folder of the test."""
    return History(tmp_path / "history.db", tmp_path / folder_name)


def test_session_with_no_turn_adds_no_conversation(tmp_path: Path) -> None:
    """Opening the history and closing without a message leaves nothing to resume."""
    assert history_of(tmp_path).conversations_in_folder() == []


def test_turns_come_back_as_they_were_kept_with_the_results_shown(tmp_path: Path) -> None:
    """Every message comes back as the model reads it, and each tool result with what the screen showed."""
    history = history_of(tmp_path)
    result = ToolResult("hello\n", "1 line", Ending.DONE)
    history.save_turn(turn_with_a_tool(), {"c1": result}, 120)
    history.save_turn(exchange("Thanks", "You are welcome."), {}, 150)
    [conversation] = history.conversations_in_folder()
    assert history.resume(conversation.id) == [
        SavedTurn(turn_with_a_tool(), {"c1": result}),
        SavedTurn(exchange("Thanks", "You are welcome."), {}),
    ]
    assert conversation.tokens_in_context == 150


def test_conversation_is_named_by_the_first_line_of_the_first_message(tmp_path: Path) -> None:
    """The title is the first line of the user's first message, cut at 60 characters."""
    long_first_line = history_of(tmp_path, "long")
    long_first_line.save_turn(exchange("x" * 70, "ok"), {}, 10)
    assert long_first_line.conversations_in_folder()[0].title == "x" * 60 + "…"
    short_first_line = history_of(tmp_path, "short")
    short_first_line.save_turn(exchange("  fix the bug\nin the loader", "ok"), {}, 10)
    assert short_first_line.conversations_in_folder()[0].title == "fix the bug"


def test_only_the_conversations_of_the_folder_are_listed_the_newest_first(tmp_path: Path) -> None:
    """Another folder's conversations stay out, and the one touched last comes first."""
    first = history_of(tmp_path)
    first.save_turn(exchange("first", "ok"), {}, 10)
    second = history_of(tmp_path)
    second.save_turn(exchange("second", "ok"), {}, 10)
    history_of(tmp_path, "other-project").save_turn(exchange("elsewhere", "ok"), {}, 10)
    first.save_turn(exchange("first again", "ok"), {}, 10)
    titles = [conversation.title for conversation in history_of(tmp_path).conversations_in_folder()]
    assert titles == ["first", "second"]


def test_turns_after_a_resume_go_to_the_conversation_resumed(tmp_path: Path) -> None:
    """A later session that continues a conversation adds its turns to it instead of starting another."""
    history_of(tmp_path).save_turn(exchange("Hi", "Hello."), {}, 10)
    later_session = history_of(tmp_path)
    [conversation] = later_session.conversations_in_folder()
    later_session.resume(conversation.id)
    later_session.save_turn(exchange("Again", "Hello again."), {}, 20)
    assert len(later_session.conversations_in_folder()) == 1
    assert [turn.messages for turn in later_session.resume(conversation.id)] == [
        exchange("Hi", "Hello."),
        exchange("Again", "Hello again."),
    ]


def test_save_that_fails_leaves_no_conversation_for_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that could not be saved adds no conversation, so the next turn starts one of its own."""
    history = history_of(tmp_path)

    def fail(*parts: object) -> None:
        """Fail as a full disk would."""
        raise OSError("no space left on the device")

    with monkeypatch.context() as patch:
        patch.setattr(queries, "insert_turn", fail)
        with pytest.raises(OSError):
            history.save_turn(exchange("Hi", "Hello."), {}, 10)
    assert history.conversation_id is None
    with transaction(tmp_path / "history.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
    history.save_turn(exchange("Hi", "Hello."), {}, 10)
    assert len(history.conversations_in_folder()) == 1
