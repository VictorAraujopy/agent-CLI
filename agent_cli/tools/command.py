"""The tool that runs a shell command in the working folder: run_command."""

import os
import signal
import subprocess
import threading
import time
from typing import Any

from agent_cli.tools.tool import Ending, Tool, ToolResult

SECONDS_A_COMMAND_MAY_RUN = 120


def start(command: str, working_folder: str) -> subprocess.Popen[str]:
    """Start the command in a shell, in a session of its own, with no input and its output in one pipe."""
    return subprocess.Popen(
        command,
        shell=True,
        cwd=working_folder,
        # The screen owns the terminal: a command that waits for input gets none and ends instead of hanging.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        # With no terminal it never draws over the screen, and ending its group ends what it started too.
        start_new_session=True,
    )


def end_group(process: subprocess.Popen[str]) -> str:
    """End the command and every process it started, and return what it wrote until then."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    output, _ = process.communicate()
    return output


def output_and_stop_reason(process: subprocess.Popen[str], stop_requested: threading.Event) -> tuple[str, str]:
    """Wait for the command and return its output, and why it was ended early: the turn stopped, or it took too long."""
    give_up_at = time.monotonic() + SECONDS_A_COMMAND_MAY_RUN
    while True:
        try:
            output, _ = process.communicate(timeout=0.1)
            return output, ""
        except subprocess.TimeoutExpired:
            pass
        if stop_requested.is_set():
            return end_group(process), "stopped by the user"
        if time.monotonic() > give_up_at:
            return end_group(process), f"stopped after {SECONDS_A_COMMAND_MAY_RUN} seconds"


def end_of(output: str) -> str:
    """Return the output, or its last 20,000 characters when it is longer, saying how many were left out."""
    most_characters = 20_000
    if len(output) <= most_characters:
        return output
    left_out = len(output) - most_characters
    return f"[the first {left_out} characters of the output were left out]\n{output[-most_characters:]}"


def last_lines(output: str) -> list[str]:
    """Return the last three lines of the output that have text."""
    lines_with_text = [line for line in output.splitlines() if line.strip()]
    return lines_with_text[-3:]


class RunCommand(Tool):
    """Runs a shell command in the working folder, after the user allows it, and returns its exit code and output."""

    name = "run_command"
    description = (
        "Run a shell command in the working folder and return its exit code and its output. "
        f"The command gets no input, and it is stopped after {SECONDS_A_COMMAND_MAY_RUN} seconds."
    )
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command, as it is typed in a shell."}},
        "required": ["command"],
    }

    def summary(self, arguments: dict[str, Any]) -> str:
        """Return the command."""
        return arguments["command"]

    def asks_permission(self, arguments: dict[str, Any]) -> bool:
        """Always ask: a command can do anything the user can."""
        return True

    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Run the command and return its exit code and output, or why it was ended before it finished."""
        process = start(arguments["command"], str(self.working_folder))
        output, stop_reason = output_and_stop_reason(process, stop_requested)
        how_it_ended = stop_reason or f"exit code {process.returncode}"
        text = f"{how_it_ended}\n{end_of(output)}"
        summary = "\n".join([how_it_ended, *last_lines(output)])
        ending = Ending.DONE if process.returncode == 0 and not stop_reason else Ending.FAILED
        return ToolResult(text, summary, ending)
