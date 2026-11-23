"""Tests for the toolbox: the tools it offers, how it reads a call the model wrote, and how it runs one."""

import json
import threading
from pathlib import Path

import pytest

from agent_cli.tools.tool import Ending, ToolError
from agent_cli.tools.toolbox import Toolbox


def test_toolbox_offers_the_file_tools_and_the_command() -> None:
    """The model is offered reading, writing and editing files, and running a command."""
    names = [schema["function"]["name"] for schema in Toolbox(Path("folder")).schemas()]
    assert names == ["read_file", "write_file", "edit_file", "run_command"]


def test_call_that_can_run_shows_what_it_acts_on_and_what_it_changes(tmp_path: Path) -> None:
    """A call read well has the summary, the preview and the question of its tool."""
    arguments = json.dumps({"path": "notes.txt", "old_text": "a", "new_text": "b"})
    use = Toolbox(tmp_path).read("edit_file", arguments)
    assert (use.summary, use.preview, use.asks_permission, use.problem) == ("notes.txt", "- a\n+ b", True, "")


@pytest.mark.parametrize(
    ("name", "arguments", "problem"),
    [
        ("delete_everything", "{}", "there is no tool delete_everything: the tools are read_file, write_file"),
        ("read_file", '{"path": "notes.txt"', "the arguments are not valid JSON"),
        ("read_file", '["notes.txt"]', "the arguments must be a JSON object"),
        ("read_file", "", "the argument path is missing"),
        ("read_file", '{"path": 7}', "the argument path must be of the type string"),
        ("read_file", '{"path": "a", "start_line": "2"}', "the argument start_line must be of the type integer"),
        ("read_file", '{"path": "a", "start_line": true}', "the argument start_line must be of the type integer"),
        ("read_file", '{"path": "~user-that-does-not-exist/a"}', "not a path that can be used"),
    ],
)
def test_call_that_cannot_run_says_why_and_shows_what_the_model_wrote(
    tmp_path: Path, name: str, arguments: str, problem: str
) -> None:
    """A call with no such tool or with wrong arguments asks nothing, and runs to an error the model reads."""
    use = Toolbox(tmp_path).read(name, arguments)
    assert problem in use.problem
    assert use.summary == arguments
    assert not use.asks_permission
    result = Toolbox(tmp_path).run(use, threading.Event())
    assert result.ending is Ending.FAILED
    assert result.text == f"error: {use.problem}"


def test_call_that_can_run_returns_the_result_of_its_tool(tmp_path: Path) -> None:
    """A good call runs its tool with its arguments."""
    (tmp_path / "notes.txt").write_text("hello\n")
    toolbox = Toolbox(tmp_path)
    result = toolbox.run(toolbox.read("read_file", '{"path": "notes.txt"}'), threading.Event())
    assert (result.text, result.ending) == ("hello\n", Ending.DONE)


@pytest.mark.parametrize("failure", [ToolError("old_text was not found"), PermissionError("permission denied")])
def test_tool_that_fails_gives_the_model_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A tool error, or an error of the system such as a file the user cannot write, is a result, not a crash."""
    toolbox = Toolbox(tmp_path)
    use = toolbox.read("read_file", '{"path": "notes.txt"}')

    def fail(*parts: object) -> None:
        """Fail as the tool would."""
        raise failure

    monkeypatch.setattr(type(use.tool), "run", fail)
    result = toolbox.run(use, threading.Event())
    assert (result.text, result.summary, result.ending) == (f"error: {failure}", str(failure), Ending.FAILED)
