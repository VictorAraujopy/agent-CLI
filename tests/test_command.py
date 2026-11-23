"""Tests for run_command: where the command runs, what it returns, and how it is ended early."""

import os
import threading
import time
from pathlib import Path

import pytest

from agent_cli.tools import command
from agent_cli.tools.command import RunCommand
from agent_cli.tools.tool import Ending, ToolResult


def run(folder: Path, shell_command: str, stop_requested: threading.Event | None = None) -> ToolResult:
    """Run the command with run_command from the folder, with a stop that is never asked for unless one is given."""
    return RunCommand(folder).run({"command": shell_command}, stop_requested or threading.Event())


def test_command_runs_in_the_working_folder(tmp_path: Path) -> None:
    """The command starts in the working folder, whatever folder the program runs in."""
    result = run(tmp_path, "pwd")
    assert result.text == f"exit code 0\n{tmp_path.resolve()}\n"
    assert result.ending is Ending.DONE


def test_output_and_errors_come_together_with_the_exit_code(tmp_path: Path) -> None:
    """What the command writes and the errors it writes come in one text, after its exit code."""
    result = run(tmp_path, "echo written; echo failed >&2; exit 3")
    assert result.text == "exit code 3\nwritten\nfailed\n"
    assert result.ending is Ending.FAILED


def test_screen_gets_the_exit_code_and_the_last_three_lines(tmp_path: Path) -> None:
    """The summary leaves out empty lines and everything but the end of the output."""
    result = run(tmp_path, "printf 'one\\ntwo\\n\\nthree\\nfour\\n'")
    assert result.summary == "exit code 0\ntwo\nthree\nfour"


def test_command_that_waits_for_input_gets_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A command that reads its input ends at once, even while the program's own input stays open, as a terminal does."""
    monkeypatch.setattr(command, "SECONDS_A_COMMAND_MAY_RUN", 3)
    read_end, write_end = os.pipe()
    input_of_the_program = os.dup(0)
    os.dup2(read_end, 0)
    try:
        result = run(tmp_path, "cat")
    finally:
        os.dup2(input_of_the_program, 0)
        for descriptor in (read_end, write_end, input_of_the_program):
            os.close(descriptor)
    assert result.text == "exit code 0\n"


def test_long_output_keeps_its_end_and_says_how_much_was_left_out(tmp_path: Path) -> None:
    """Only the last 20,000 characters reach the model, where the errors and the summary of a run usually are."""
    result = run(tmp_path, "head -c 30000 /dev/zero | tr '\\0' 'x'; echo; echo the-end")
    assert "[the first 10009 characters of the output were left out]" in result.text
    assert result.text.endswith("the-end\n")


def test_stop_ends_the_command_and_what_it_started(tmp_path: Path) -> None:
    """A stop of the turn ends the command at once, children included, and keeps what it wrote."""
    stop_requested = threading.Event()
    marker = tmp_path / "child-pid"
    threading.Timer(0.5, stop_requested.set).start()
    started = time.monotonic()
    result = run(tmp_path, f"echo before; sleep 30 & echo $! > {marker}; wait", stop_requested)
    assert time.monotonic() - started < 5
    assert result.text == "stopped by the user\nbefore\n"
    assert result.ending is Ending.FAILED
    child = int(marker.read_text())
    give_up_at = time.monotonic() + 5
    while time.monotonic() < give_up_at and child_is_alive(child):
        time.sleep(0.05)
    assert not child_is_alive(child)


def child_is_alive(pid: int) -> bool:
    """Tell whether a process of that number still runs and is not a zombie waiting to be collected."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = os.popen(f"ps -o state= -p {pid}").read().strip()
    return state != "" and not state.startswith("Z")


def test_command_that_takes_too_long_is_ended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A command that passes the time allowed is ended, and the model is told why."""
    monkeypatch.setattr(command, "SECONDS_A_COMMAND_MAY_RUN", 0.5)
    started = time.monotonic()
    result = run(tmp_path, "sleep 30")
    assert time.monotonic() - started < 5
    assert result.text == "stopped after 0.5 seconds\n"


def test_running_a_command_always_asks(tmp_path: Path) -> None:
    """A command can do anything the user can, so it always waits for the user to allow it."""
    assert RunCommand(tmp_path).asks_permission({"command": "ls"})
    assert RunCommand(tmp_path).summary({"command": "ls -la"}) == "ls -la"
