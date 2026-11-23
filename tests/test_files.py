"""Tests for the file tools: what read_file returns, what write_file and edit_file change, and when they ask first."""

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_cli.tools.files import EditFile, ReadFile, WriteFile
from agent_cli.tools.tool import ToolError, ToolResult


def read(folder: Path, path: str, start_line: int | None = None) -> ToolResult:
    """Run read_file from the folder on the path, from the line given, if any."""
    arguments: dict[str, object] = {"path": path}
    if start_line is not None:
        arguments["start_line"] = start_line
    return ReadFile(folder).run(arguments, threading.Event())


def numbered_lines(count: int) -> str:
    """Return a text of that many short lines, each saying its number."""
    return "".join(f"line {number}\n" for number in range(1, count + 1))


def test_short_file_is_read_whole(tmp_path: Path) -> None:
    """A file that fits in one part comes as it is, and the screen is told its lines."""
    (tmp_path / "notes.txt").write_text("one\ntwo\n")
    result = read(tmp_path, "notes.txt")
    assert result.text == "one\ntwo\n"
    assert result.summary == "2 lines"


def test_empty_file_says_it_is_empty(tmp_path: Path) -> None:
    """An empty file gives a text that says so, instead of nothing the model could misread."""
    (tmp_path / "empty.txt").write_text("")
    assert read(tmp_path, "empty.txt").text == "(the file is empty)"


def test_long_file_comes_in_parts_of_500_lines_that_say_how_to_read_on(tmp_path: Path) -> None:
    """The first part ends with the range shown and the line to start the next call at."""
    (tmp_path / "long.txt").write_text(numbered_lines(1200))
    result = read(tmp_path, "long.txt")
    assert result.text.startswith("line 1\n")
    assert "line 500\n" in result.text
    assert "line 501\n" not in result.text
    assert result.text.endswith("[lines 1-500 of 1200. To read on, call read_file with start_line=501.]")
    assert result.summary == "lines 1-500 of 1200"


def test_last_part_says_it_reaches_the_end(tmp_path: Path) -> None:
    """A part that starts further down and reaches the last line says it is the end of the file."""
    (tmp_path / "long.txt").write_text(numbered_lines(1200))
    result = read(tmp_path, "long.txt", start_line=1001)
    assert result.text.startswith("line 1001\n")
    assert result.text.endswith("line 1200\n\n[lines 1001-1200 of 1200: the end of the file.]")


def test_part_stops_at_40000_characters(tmp_path: Path) -> None:
    """Long lines end a part before 500 of them, so a part never takes more than 40,000 characters."""
    (tmp_path / "wide.txt").write_text(("x" * 999 + "\n") * 100)
    result = read(tmp_path, "wide.txt")
    assert result.summary == "lines 1-40 of 100"


def test_single_line_longer_than_a_part_comes_cut_and_says_so(tmp_path: Path) -> None:
    """A minified file of one huge line does not reach the model whole."""
    (tmp_path / "minified.js").write_text("x" * 100_000)
    result = read(tmp_path, "minified.js")
    assert result.text.count("x") == 40_000
    assert "[this line goes on: cut at 40000 characters]" in result.text


def test_start_line_past_the_end_is_refused_with_the_lines_there_are(tmp_path: Path) -> None:
    """A start line outside the file tells the model how many lines the file has."""
    (tmp_path / "notes.txt").write_text("one\ntwo\n")
    with pytest.raises(ToolError, match="between 1 and 2"):
        read(tmp_path, "notes.txt", start_line=3)
    with pytest.raises(ToolError, match="between 1 and 2"):
        read(tmp_path, "notes.txt", start_line=0)


@pytest.mark.parametrize("make", [lambda path: None, lambda path: path.mkdir()])
def test_missing_file_or_folder_is_not_read(tmp_path: Path, make: Callable[[Path], object]) -> None:
    """Only a file can be read: nothing there, or a folder, is a tool error."""
    make(tmp_path / "notes.txt")
    with pytest.raises(ToolError, match="no file at"):
        read(tmp_path, "notes.txt")


def test_file_that_is_not_text_is_not_read(tmp_path: Path) -> None:
    """Bytes that are not UTF-8 text are a tool error, not a broken turn."""
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    with pytest.raises(ToolError, match="not a text file"):
        read(tmp_path, "image.png")


def test_reading_asks_only_for_a_secret_wherever_it_is(tmp_path: Path) -> None:
    """Any file reads at once, in the folder or out of it; a key or a .env asks, also when a link reaches it."""
    working_folder = tmp_path / "project"
    working_folder.mkdir()
    (tmp_path / ".ssh").mkdir()
    (working_folder / "keys").symlink_to(tmp_path / ".ssh")
    tool = ReadFile(working_folder)
    assert not tool.asks_permission({"path": "notes.txt"})
    assert not tool.asks_permission({"path": "/etc/hosts"})
    assert not tool.asks_permission({"path": "../notes.txt"})
    assert tool.asks_permission({"path": "~/.ssh/id_ed25519"})
    assert tool.asks_permission({"path": "keys/id_ed25519"})
    assert tool.asks_permission({"path": ".env"})
    assert tool.asks_permission({"path": "config/.env.local"})
    assert tool.asks_permission({"path": "~/.netrc"})
    assert tool.asks_permission({"path": "~/.aws/credentials"})
    assert tool.asks_permission({"path": "~/.gnupg/private-keys-v1.d/key"})
    assert tool.asks_permission({"path": "~/.agent-cli/history.db"})


def test_write_creates_the_file_and_the_folders_on_its_way(tmp_path: Path) -> None:
    """A new file is created with its folders, and the result says it was created."""
    result = WriteFile(tmp_path).run({"path": "a/b/new.py", "content": "print(1)\n"}, threading.Event())
    assert (tmp_path / "a" / "b" / "new.py").read_text() == "print(1)\n"
    assert result.summary == "created, 1 line"


def test_write_replaces_the_whole_content_of_a_file(tmp_path: Path) -> None:
    """An existing file gets only the new content, and the result says it was replaced."""
    (tmp_path / "old.txt").write_text("old\ncontent\n")
    result = WriteFile(tmp_path).run({"path": "old.txt", "content": "new\n"}, threading.Event())
    assert (tmp_path / "old.txt").read_text() == "new\n"
    assert result.summary == "replaced, 1 line"


def test_write_to_a_folder_is_refused(tmp_path: Path) -> None:
    """A path that is a folder is not turned into a file."""
    (tmp_path / "folder").mkdir()
    with pytest.raises(ToolError, match="is a folder"):
        WriteFile(tmp_path).run({"path": "folder", "content": "x"}, threading.Event())


def test_write_shows_the_first_lines_it_brings_and_how_many_more(tmp_path: Path) -> None:
    """The preview marks each line with +, and a long content shows eight lines and the count of the rest."""
    preview = WriteFile(tmp_path).preview({"path": "new.txt", "content": numbered_lines(10)})
    assert preview.splitlines() == [f"+ line {number}" for number in range(1, 9)] + ["  … 2 more lines"]


def test_edit_replaces_the_one_place_of_the_old_text(tmp_path: Path) -> None:
    """The old text is replaced where it is, and the rest of the file stays as it was."""
    (tmp_path / "notes.txt").write_text("Thsi is a line.\nAnother line.\n")
    arguments = {"path": "notes.txt", "old_text": "Thsi", "new_text": "This"}
    result = EditFile(tmp_path).run(arguments, threading.Event())
    assert (tmp_path / "notes.txt").read_text() == "This is a line.\nAnother line.\n"
    assert result.summary == "1 line replaced by 1 line"


@pytest.mark.parametrize(("old_text", "reason"), [("missing", "was not found"), ("", "was not found"), ("line", "2 times")])
def test_edit_refuses_old_text_found_in_no_place_or_in_more_than_one(tmp_path: Path, old_text: str, reason: str) -> None:
    """Old text that is not in the file, is empty, or is in it twice, changes nothing and says why."""
    (tmp_path / "notes.txt").write_text("a line\nanother line\n")
    with pytest.raises(ToolError, match=reason):
        EditFile(tmp_path).run({"path": "notes.txt", "old_text": old_text, "new_text": "x"}, threading.Event())
    assert (tmp_path / "notes.txt").read_text() == "a line\nanother line\n"


def test_edit_shows_the_lines_that_leave_and_the_ones_that_come(tmp_path: Path) -> None:
    """The preview marks the old lines with - and the new ones with +."""
    arguments = {"path": "notes.txt", "old_text": "a\nb", "new_text": "c"}
    assert EditFile(tmp_path).preview(arguments).splitlines() == ["- a", "- b", "+ c"]


@pytest.mark.parametrize("tool", [WriteFile, EditFile])
def test_tools_that_change_a_file_always_ask(tmp_path: Path, tool: type[WriteFile] | type[EditFile]) -> None:
    """A change asks even inside the working folder."""
    assert tool(tmp_path).asks_permission({"path": "notes.txt", "content": "", "old_text": "", "new_text": ""})
