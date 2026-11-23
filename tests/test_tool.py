"""Tests for what every tool shares: the paths a call names, the count of lines, and how a tool is offered."""

from pathlib import Path

import pytest

from agent_cli.tools.files import ReadFile
from agent_cli.tools.tool import ToolError, lines_in, path_in


def test_relative_path_starts_at_the_working_folder(tmp_path: Path) -> None:
    """A path written without a root is taken from the working folder, not from the folder the program runs in."""
    assert path_in(tmp_path, "folder/notes.txt") == tmp_path.resolve() / "folder" / "notes.txt"


def test_absolute_path_and_home_folder_are_kept(tmp_path: Path) -> None:
    """An absolute path stays as written, and ~ becomes the home folder."""
    assert path_in(tmp_path, "/etc/hosts") == Path("/etc/hosts").resolve()
    assert path_in(tmp_path, "~/notes.txt") == Path.home().resolve() / "notes.txt"


def test_path_that_cannot_be_used_is_a_tool_error(tmp_path: Path) -> None:
    """An unknown user after ~ or a null character tell the model the path is wrong, instead of breaking the turn."""
    with pytest.raises(ToolError, match="not a path that can be used"):
        path_in(tmp_path, "~user-that-does-not-exist/notes.txt")
    with pytest.raises(ToolError, match="not a path that can be used"):
        path_in(tmp_path, "notes\0.txt")


@pytest.mark.parametrize(("text", "said"), [("", "0 lines"), ("one", "1 line"), ("one\ntwo\n", "2 lines")])
def test_lines_are_counted_in_words(text: str, said: str) -> None:
    """The count of lines reads as words, with one line in the singular."""
    assert lines_in(text) == said


def test_tool_is_offered_in_the_chat_completions_format(tmp_path: Path) -> None:
    """A tool goes to the model as a function with its name, its description and the schema of its arguments."""
    tool = ReadFile(tmp_path)
    assert tool.schema() == {
        "type": "function",
        "function": {"name": "read_file", "description": tool.description, "parameters": tool.parameters},
    }
