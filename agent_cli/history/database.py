"""The file of the history: created once, readable only by the user, and opened for each unit of work."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec


def create_database(database_file: Path) -> None:
    """Create the file, its folder and its tables when they are missing; only the user can read the conversations."""
    database_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    database_file.touch(mode=0o600, exist_ok=True)
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with transaction(database_file) as connection:
        connection.executescript(schema)


def load_vector_functions(connection: sqlite3.Connection) -> None:
    """Teach the connection to compare vectors, then refuse any other extension a query could ask to load."""
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)


@contextmanager
def transaction(database_file: Path) -> Iterator[sqlite3.Connection]:
    """Open the database for one unit of work, saved whole at its end or not at all on an error, and close it."""
    # A connection of its own for each unit of work: the turns are saved from the thread of the turn, and a
    # connection of SQLite only works in the thread that opened it.
    connection = sqlite3.connect(database_file)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()
