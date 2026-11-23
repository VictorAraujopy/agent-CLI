"""The tool that runs a shell command in the working folder: run_command."""

import glob
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agent_cli.tools.tool import Ending, Tool, ToolError, ToolResult, path_in

SECONDS_A_COMMAND_MAY_RUN = 120

# The commands that only read, each with the options that would make it run a program, write a file or follow
# links out of the folder.
READING_COMMANDS = {
    "pwd": (),
    "ls": (),
    "cat": (),
    "head": (),
    "tail": (),
    "wc": (),
    "grep": ("-R", "-S", "--dereference-recursive"),
    "rg": ("-L", "--follow", "--pre", "--hostname-bin"),
    "find": (
        "-L", "-follow", "-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf"
    ),
}


def words_of(command: str) -> list[str] | None:
    """Split the command into words and operators as the shell does; None when its quotes are not closed."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    # The shell only takes # as a comment at the start of a word: kept as text, nothing after it is hidden.
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def without_redirects_to_nowhere(words: list[str]) -> list[str]:
    """Return the words without each redirect to /dev/null, which only throws output away."""
    kept: list[str] = []
    for word in words:
        if word == "/dev/null" and kept and kept[-1] in (">", ">>"):
            kept.pop()
            continue
        kept.append(word)
    return kept


def is_operator(word: str) -> bool:
    """Tell whether the word is an operator of the shell, such as ; or && or >: made only of their characters."""
    return all(character in "();<>|&" for character in word)


def commands_in_the_pipe(words: list[str]) -> list[list[str]]:
    """Split the words at each pipe into the commands the pipe joins."""
    commands: list[list[str]] = [[]]
    for word in words:
        if word == "|":
            commands.append([])
        else:
            commands[-1].append(word)
    return commands


def turns_on(word: str, option: str) -> bool:
    """Tell whether the word turns the option on: whole, abbreviated, with a value or among other one-letter options."""
    if option.startswith("--"):
        name = word.split("=")[0]
        return len(name) > 2 and option.startswith(name)
    if len(option) == 2:
        return word.startswith("-") and not word.startswith("--") and option[1] in word[1:]
    return word == option


def stays_in_the_folder(word: str, working_folder: Path) -> bool:
    """Tell whether the word, taken as a path, and each path its wildcards match, stay inside the working folder."""
    written = os.path.expanduser(word)
    # A link matched by a wildcard would take the command out of the folder with no path saying so.
    paths = [written, *glob.glob(written, root_dir=working_folder)]
    try:
        return all(path_in(working_folder, path).is_relative_to(working_folder) for path in paths)
    except ToolError:
        return False


def only_reads_in_the_folder(words: list[str], working_folder: Path) -> bool:
    """Tell whether one command of the pipe is a reading command, with no option it forbids and no path outside."""
    if not words or words[0] not in READING_COMMANDS:
        return False
    forbidden_options = READING_COMMANDS[words[0]]
    for word in words[1:]:
        if any(turns_on(word, option) for option in forbidden_options):
            return False
        is_option = word.startswith("-")
        # The path in the value of an option is not looked at, so an option that holds one asks.
        if is_option and "/" in word:
            return False
        if not is_option and not stays_in_the_folder(word, working_folder):
            return False
    return True


def only_reads(command: str, working_folder: Path) -> bool:
    """Tell whether the command only reads what is in the working folder, so it can run without asking."""
    # $ and ` run a command even between double quotes, { } turn into paths no word shows, and a new line
    # starts another command where the split sees a space.
    if any(character in "$`{}\n\r" for character in command):
        return False
    words = words_of(command)
    if words is None:
        return False
    words = without_redirects_to_nowhere(words)
    if any(is_operator(word) and word != "|" for word in words):
        return False
    return all(only_reads_in_the_folder(each, working_folder) for each in commands_in_the_pipe(words))


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
    """Runs a shell command in the working folder, asking first unless it only reads, and returns its output."""

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
        """Ask unless the command only reads what is in the working folder: any other can do anything the user can."""
        return not only_reads(arguments["command"], self.working_folder)

    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Run the command and return its exit code and output, or why it was ended before it finished."""
        process = start(arguments["command"], str(self.working_folder))
        output, stop_reason = output_and_stop_reason(process, stop_requested)
        how_it_ended = stop_reason or f"exit code {process.returncode}"
        text = f"{how_it_ended}\n{end_of(output)}"
        summary = "\n".join([how_it_ended, *last_lines(output)])
        ending = Ending.DONE if process.returncode == 0 and not stop_reason else Ending.FAILED
        return ToolResult(text, summary, ending)
