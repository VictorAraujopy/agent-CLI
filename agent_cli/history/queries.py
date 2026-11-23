"""The queries of the history: each turn saved as it ends, and the conversations of a folder read back."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_cli.history.database import create_database, transaction
from agent_cli.inference.model import Message
from agent_cli.tools.tool import Ending, ToolResult


@dataclass(frozen=True)
class SavedConversation:
    """A conversation of the history, as the list to resume it shows it."""

    id: int
    title: str
    updated_at: str
    tokens_in_context: int


@dataclass(frozen=True)
class SavedTurn:
    """A turn read back: its messages as the model reads them, and what the screen showed of each tool result."""

    messages: list[Message]
    tool_results: dict[str, ToolResult]


def now() -> str:
    """Return the time now, to the second, as the history writes it."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def title_of(messages: list[Message]) -> str:
    """Return the name of a conversation: the first line of the user's first message, cut at 60 characters."""
    first_line = messages[0]["content"].strip().split("\n")[0]
    if len(first_line) > 60:
        return f"{first_line[:60]}…"
    return first_line


def insert_conversation(connection: sqlite3.Connection, working_folder: str, title: str) -> int:
    """Add a conversation of the working folder, with no turn yet, and return its id."""
    started_at = now()
    cursor = connection.execute(
        "INSERT INTO conversations (working_folder, title, started_at, updated_at, tokens_in_context)"
        " VALUES (?, ?, ?, ?, 0)",
        (working_folder, title, started_at, started_at),
    )
    return cursor.lastrowid


def row_of(message: Message, tool_results: dict[str, ToolResult]) -> dict[str, object]:
    """Return the columns that keep a message, with what the screen showed of it when it is a tool result."""
    row: dict[str, object] = {
        "role": message["role"],
        "content": message["content"],
        "tool_calls": None,
        "tool_call_id": message.get("tool_call_id"),
        "tool_summary": None,
        "tool_ending": None,
    }
    if "tool_calls" in message:
        row["tool_calls"] = json.dumps(message["tool_calls"])
    result = tool_results.get(message.get("tool_call_id", ""))
    if result is not None:
        row["tool_summary"] = result.summary
        row["tool_ending"] = result.ending.value
    return row


def insert_turn(
    connection: sqlite3.Connection, conversation_id: int, messages: list[Message], tool_results: dict[str, ToolResult]
) -> None:
    """Add the messages of the next turn of the conversation, with what the screen showed of each tool result."""
    last_turn = connection.execute(
        "SELECT COALESCE(MAX(turn), 0) FROM messages WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()[0]
    for message in messages:
        row = {"conversation_id": conversation_id, "turn": last_turn + 1, **row_of(message, tool_results)}
        connection.execute(
            "INSERT INTO messages (conversation_id, turn, role, content, tool_calls, tool_call_id, tool_summary,"
            " tool_ending) VALUES (:conversation_id, :turn, :role, :content, :tool_calls, :tool_call_id,"
            " :tool_summary, :tool_ending)",
            row,
        )


def message_of(row: sqlite3.Row) -> Message:
    """Return a message as it was kept in the conversation, from its row."""
    message: Message = {"role": row["role"], "content": row["content"]}
    if row["tool_calls"] is not None:
        message["tool_calls"] = json.loads(row["tool_calls"])
    if row["tool_call_id"] is not None:
        message["tool_call_id"] = row["tool_call_id"]
    return message


def turns_in(rows: list[sqlite3.Row]) -> list[SavedTurn]:
    """Group the rows of a conversation, in order, into its turns."""
    turns: dict[int, SavedTurn] = {}
    for row in rows:
        turn = turns.setdefault(row["turn"], SavedTurn([], {}))
        turn.messages.append(message_of(row))
        if row["tool_ending"] is not None:
            ending = Ending(row["tool_ending"])
            turn.tool_results[row["tool_call_id"]] = ToolResult(row["content"], row["tool_summary"], ending)
    return list(turns.values())


class History:
    """The history of the conversations held in one file, and the conversation the turns of this session go to."""

    def __init__(self, database_file: Path, working_folder: Path) -> None:
        """Create the file if it is missing; the conversation of this session is added at its first turn."""
        create_database(database_file)
        self.database_file = database_file
        self.working_folder = str(working_folder.resolve())
        self.conversation_id: int | None = None

    def save_turn(self, messages: list[Message], tool_results: dict[str, ToolResult], tokens_in_context: int) -> None:
        """Save a turn that ended, in the conversation of this session, which the first turn adds."""
        with transaction(self.database_file) as connection:
            conversation_id = self.conversation_id
            if conversation_id is None:
                conversation_id = insert_conversation(connection, self.working_folder, title_of(messages))
            insert_turn(connection, conversation_id, messages, tool_results)
            connection.execute(
                "UPDATE conversations SET updated_at = ?, tokens_in_context = ? WHERE id = ?",
                (now(), tokens_in_context, conversation_id),
            )
        # Only after the turn is saved: a failed save leaves no conversation behind for the next turn to use.
        self.conversation_id = conversation_id

    def conversations_in_folder(self) -> list[SavedConversation]:
        """Return the conversations of the working folder, the one with the turn saved last first."""
        # By the last message saved, not by the time: two turns saved in the same second would tie.
        with transaction(self.database_file) as connection:
            rows = connection.execute(
                "SELECT conversations.id, title, updated_at, tokens_in_context FROM conversations"
                " JOIN messages ON messages.conversation_id = conversations.id"
                " WHERE working_folder = ? GROUP BY conversations.id ORDER BY MAX(messages.id) DESC",
                (self.working_folder,),
            ).fetchall()
        return [SavedConversation(row["id"], row["title"], row["updated_at"], row["tokens_in_context"]) for row in rows]

    def resume(self, conversation_id: int) -> list[SavedTurn]:
        """Make the conversation the one the next turns go to, and return its turns."""
        with transaction(self.database_file) as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        self.conversation_id = conversation_id
        return turns_in(rows)
