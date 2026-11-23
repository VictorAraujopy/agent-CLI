"""Tests for the file of the history: who can read it, its tables, and how a unit of work is saved."""

import sqlite3
import stat
from pathlib import Path

import pytest

from agent_cli.history.database import create_database, transaction


def permissions_of(path: Path) -> int:
    """Return the permission bits of a file or folder."""
    return stat.S_IMODE(path.stat().st_mode)


def test_file_and_folder_are_created_readable_only_by_the_user(tmp_path: Path) -> None:
    """The conversations are private: the folder is 700 and the file 600."""
    database_file = tmp_path / "agent-cli" / "history.db"
    create_database(database_file)
    assert permissions_of(database_file.parent) == 0o700
    assert permissions_of(database_file) == 0o600


def test_tables_are_created_once_and_kept(tmp_path: Path) -> None:
    """Creating the database of a file that has one keeps what it holds."""
    database_file = tmp_path / "history.db"
    create_database(database_file)
    with transaction(database_file) as connection:
        connection.execute("INSERT INTO conversations VALUES (1, '/folder', 'title', 'then', 'then', 0)")
    create_database(database_file)
    with transaction(database_file) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1


def test_unit_of_work_that_fails_saves_nothing_and_closes(tmp_path: Path) -> None:
    """An error in the middle undoes what the unit of work did, and the connection is closed either way."""
    database_file = tmp_path / "history.db"
    create_database(database_file)
    with pytest.raises(RuntimeError), transaction(database_file) as connection:
        connection.execute("INSERT INTO conversations VALUES (1, '/folder', 'title', 'then', 'then', 0)")
        raise RuntimeError("failed halfway")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with transaction(database_file) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
