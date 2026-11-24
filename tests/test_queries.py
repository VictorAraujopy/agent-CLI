"""Tests for the queries of the history: turns saved as they end, and conversations read back to continue them."""

import json
import math
import sqlite3
from pathlib import Path

import pytest

from agent_cli.history import queries
from agent_cli.history.database import load_vector_functions, transaction
from agent_cli.history.queries import History, SavedExcerpt, SavedTurn, TurnToIndex
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


def excerpt_for(turn: TurnToIndex, vector: list[float]) -> SavedExcerpt:
    """Return an excerpt of the turn, named after its place, with the vector given."""
    return SavedExcerpt(turn.conversation_id, turn.turn, f"turn {turn.turn}", vector)


def test_turns_to_index_are_those_of_the_folder_the_model_made_no_vector_of(tmp_path: Path) -> None:
    """Each turn of the folder comes once, without its tool results, until the embedding model indexes it."""
    history = history_of(tmp_path)
    history.save_turn(turn_with_a_tool(), {"c1": ToolResult("hello\n", "1 line", Ending.DONE)}, 10)
    history.save_turn(exchange("Thanks", "You are welcome."), {}, 20)
    history_of(tmp_path, "other-project").save_turn(exchange("elsewhere", "ok"), {}, 10)
    first, second = history.turns_with_no_excerpt("author/embedding")
    assert (first.turn, second.turn) == (1, 2)
    assert first.messages == [message for message in turn_with_a_tool() if message["role"] != "tool"]
    assert second.messages == exchange("Thanks", "You are welcome.")
    history.save_excerpts("author/embedding", [excerpt_for(first, [1.0])])
    assert history.turns_with_no_excerpt("author/embedding") == [second]
    assert len(history.turns_with_no_excerpt("another/embedding")) == 2


def closest_to_east(history: History, turns_in_context: int = 0, how_many: int = 10) -> list[str]:
    """Return the excerpts of the folder by author/embedding above 0.6 of similarity to the vector [1, 0]."""
    return history.excerpts_closest_to("author/embedding", [1.0, 0.0], turns_in_context, 0.6, how_many)


def at_similarity(cosine: float) -> list[float]:
    """Return a vector whose similarity to [1, 0] is the cosine given."""
    return [cosine, math.sqrt(1 - cosine**2)]


def test_closest_excerpts_above_the_minimum_come_closest_first_up_to_how_many(tmp_path: Path) -> None:
    """The database compares the vectors and returns only the closest, from the closest, and only above the minimum."""
    history = history_of(tmp_path)
    cosines = [0.7, 0.95, 0.59, 0.8, 0.61]
    for cosine in cosines:
        history.save_turn(exchange(f"at {cosine}", "ok"), {}, 10)
    turns = history.turns_with_no_excerpt("author/embedding")
    excerpts = [
        SavedExcerpt(turn.conversation_id, turn.turn, f"at {cosine}", at_similarity(cosine))
        for turn, cosine in zip(turns, cosines)
    ]
    history.save_excerpts("author/embedding", excerpts)
    session = history_of(tmp_path)
    assert closest_to_east(session, how_many=3) == ["at 0.95", "at 0.8", "at 0.7"]
    assert closest_to_east(session, how_many=10) == ["at 0.95", "at 0.8", "at 0.7", "at 0.61"]


def test_session_with_no_turn_saved_yet_searches_every_earlier_conversation(tmp_path: Path) -> None:
    """A session that has no conversation of its own yet leaves nothing out of the search."""
    earlier_session = history_of(tmp_path)
    earlier_session.save_turn(exchange("Hi", "Hello."), {}, 10)
    [turn] = earlier_session.turns_with_no_excerpt("author/embedding")
    earlier_session.save_excerpts("author/embedding", [excerpt_for(turn, [1.0, 0.0])])
    assert closest_to_east(history_of(tmp_path), turns_in_context=0) == ["turn 1"]


def test_excerpts_of_the_turns_the_model_still_reads_are_left_out(tmp_path: Path) -> None:
    """Of the conversation of the session, only the turns before those in the context come back; others come whole."""
    earlier_session = history_of(tmp_path)
    earlier_session.save_turn(exchange("earlier", "ok"), {}, 10)
    session = history_of(tmp_path)
    for question in ("one", "two", "three"):
        session.save_turn(exchange(question, "ok"), {}, 10)
    turns = session.turns_with_no_excerpt("author/embedding")
    texts = ["earlier", "one", "two", "three"]
    vectors = [at_similarity(0.7), at_similarity(0.8), at_similarity(0.9), at_similarity(0.95)]
    excerpts = [
        SavedExcerpt(turn.conversation_id, turn.turn, text, vector)
        for turn, text, vector in zip(turns, texts, vectors)
    ]
    session.save_excerpts("author/embedding", excerpts)
    assert closest_to_east(session, turns_in_context=2) == ["one", "earlier"]
    assert closest_to_east(session, turns_in_context=0) == ["three", "two", "one", "earlier"]


def test_turns_left_out_take_no_place_among_the_closest(tmp_path: Path) -> None:
    """The turns in the context leave before the closest are counted, so the closest of the rest still come."""
    session = history_of(tmp_path)
    for question in ("old", "in the context"):
        session.save_turn(exchange(question, "ok"), {}, 10)
    old, in_the_context = session.turns_with_no_excerpt("author/embedding")
    session.save_excerpts(
        "author/embedding",
        [
            SavedExcerpt(old.conversation_id, old.turn, "old", at_similarity(0.7)),
            SavedExcerpt(in_the_context.conversation_id, in_the_context.turn, "in the context", at_similarity(0.99)),
        ],
    )
    assert closest_to_east(session, turns_in_context=1, how_many=1) == ["old"]


def test_excerpts_of_another_folder_or_another_embedding_model_are_left_out(tmp_path: Path) -> None:
    """Only the vectors of this folder made by this embedding model can be compared with the message."""
    elsewhere = history_of(tmp_path, "other-project")
    elsewhere.save_turn(exchange("elsewhere", "ok"), {}, 10)
    [turn_elsewhere] = elsewhere.turns_with_no_excerpt("author/embedding")
    elsewhere.save_excerpts("author/embedding", [excerpt_for(turn_elsewhere, [1.0, 0.0])])
    here = history_of(tmp_path)
    here.save_turn(exchange("here", "ok"), {}, 10)
    [turn_here] = here.turns_with_no_excerpt("another/embedding")
    here.save_excerpts("another/embedding", [excerpt_for(turn_here, [1.0, 0.0])])
    assert closest_to_east(here) == []


def test_turn_indexed_twice_keeps_one_excerpt(tmp_path: Path) -> None:
    """Two sessions of the same folder that index the same turn at once leave one excerpt of it, and no error."""
    history = history_of(tmp_path)
    history.save_turn(exchange("Hi", "Hello."), {}, 10)
    [turn] = history.turns_with_no_excerpt("author/embedding")
    history.save_excerpts("author/embedding", [excerpt_for(turn, [1.0, 0.0])])
    history.save_excerpts("author/embedding", [excerpt_for(turn, [1.0, 0.0])])
    assert closest_to_east(history_of(tmp_path)) == ["turn 1"]


def test_connections_other_than_the_search_load_no_extension(tmp_path: Path) -> None:
    """Keeping and reading back the history needs no extension, and its connections cannot load one."""
    history_of(tmp_path)
    with transaction(tmp_path / "history.db") as connection:
        with pytest.raises(sqlite3.OperationalError, match="no such function: vec_version"):
            connection.execute("SELECT vec_version()")
        with pytest.raises(sqlite3.OperationalError, match="not authorized"):
            connection.execute("SELECT load_extension('anything')")


def test_connection_of_the_search_compares_vectors_and_then_loads_nothing_else(tmp_path: Path) -> None:
    """The vector functions are loaded, and the loading of extensions is turned off right after."""
    history_of(tmp_path)
    with transaction(tmp_path / "history.db") as connection:
        load_vector_functions(connection)
        assert connection.execute("SELECT vec_distance_cosine('[1, 0]', '[0, 1]')").fetchone()[0] == 1.0
        with pytest.raises(sqlite3.OperationalError, match="not authorized"):
            connection.execute("SELECT load_extension('anything')")


def test_history_works_where_sqlite_cannot_load_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a Python built without extension loading, only the search fails; the turns are kept and read back."""

    def fail(connection: sqlite3.Connection) -> None:
        """Fail as a Python whose SQLite cannot load extensions does."""
        raise AttributeError("'sqlite3.Connection' object has no attribute 'enable_load_extension'")

    monkeypatch.setattr(queries, "load_vector_functions", fail)
    history = history_of(tmp_path)
    history.save_turn(exchange("Hi", "Hello."), {}, 10)
    [turn] = history.turns_with_no_excerpt("author/embedding")
    history.save_excerpts("author/embedding", [excerpt_for(turn, [1.0, 0.0])])
    [conversation] = history.conversations_in_folder()
    assert history.resume(conversation.id)[0].messages == exchange("Hi", "Hello.")
    with pytest.raises(AttributeError, match="enable_load_extension"):
        closest_to_east(history)


def test_history_saved_before_the_search_existed_gets_its_table_and_its_turns_indexed(tmp_path: Path) -> None:
    """A file made before the excerpts existed gains their table when it is opened, and its turns can be indexed."""
    current_schema = Path(queries.__file__).with_name("schema.sql").read_text()
    schema_before_the_excerpts = current_schema.split("CREATE TABLE IF NOT EXISTS excerpts")[0]
    working_folder = str((tmp_path / "project").resolve())
    with sqlite3.connect(tmp_path / "history.db") as connection:
        connection.executescript(schema_before_the_excerpts)
        connection.execute("INSERT INTO conversations VALUES (1, ?, 'Hi', 'now', 'now', 10)", (working_folder,))
        connection.execute("INSERT INTO messages (conversation_id, turn, role, content) VALUES (1, 1, 'user', 'Hi')")
    connection.close()
    [turn] = history_of(tmp_path).turns_with_no_excerpt("author/embedding")
    assert turn.messages == [{"role": "user", "content": "Hi"}]
