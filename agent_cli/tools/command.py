"""The tool that runs a shell command in the working folder, run_command, and how it tells a command that only reads."""

import glob
import os
import shlex
import signal
import subprocess
import threading
import time
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from agent_cli.tools.tool import Ending, Tool, ToolError, ToolResult, is_secret, path_in

SECONDS_A_COMMAND_MAY_RUN = 120

# The commands that only read, each with the options that would make it run a program, write a file, or follow
# links inside the folders it searches, where the look for secrets does not go.
READING_COMMANDS = {
    "pwd": (),
    "ls": (),
    "cat": (),
    "head": (),
    "tail": (),
    "wc": (),
    "du": (),
    "mdfind": (),
    "grep": ("-R", "-S", "--dereference-recursive"),
    "rg": ("-L", "--follow", "--pre", "--hostname-bin"),
    "find": ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf"),
}

# A search inside files looks through its folders for a secret first; a folder with more entries than this asks.
ENTRIES_A_SEARCH_MAY_LOOK_THROUGH = 20_000

# The operators that only join commands: with each command checked on its own, they read no more than one would.
COMMAND_SEPARATORS = ("|", "&&", "||", ";")

# A lone parenthesis groups the tests of find, as in \( -name a -o -name b \). Where it runs something, a subshell at
# the start of a command or a function right after its name, reads_no_secret refuses the command.
GROUPING_PARENTHESES = ("(", ")")


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


def without_harmless_redirects(words: list[str]) -> list[str]:
    """Return the words without the redirects that write no file: to /dev/null, or into another output, as 2>&1."""
    kept: list[str] = []
    for word in words:
        previous = kept[-1] if kept else ""
        to_nowhere = word == "/dev/null" and previous in (">", ">>", "&>", "&>>")
        # >& followed by a name, not a number, writes both outputs to that file.
        into_another_output = word.isdigit() and previous == ">&"
        if to_nowhere or into_another_output:
            kept.pop()
            continue
        kept.append(word)
    return kept


def is_operator(word: str) -> bool:
    """Tell whether the word is an operator of the shell, such as ; or && or >: made only of their characters."""
    return all(character in "();<>|&" for character in word)


def commands_in(words: list[str]) -> list[list[str]]:
    """Split the words into the commands that |, &&, || and ; join."""
    commands: list[list[str]] = [[]]
    for word in words:
        if word in COMMAND_SEPARATORS:
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


def value_of(option: str) -> str:
    """Return what an option holds: what follows its =, or the letters after a short one, as in -f.env."""
    if "=" in option:
        return option.split("=", 1)[1]
    if option.startswith("--"):
        return ""
    return option[2:]


def reaches_no_secret(path_written: str, working_folder: Path) -> bool:
    """Tell whether the path, and each path its wildcards match, stay out of the secret places, links followed."""
    expanded = os.path.expanduser(path_written)
    # A wildcard can match a secret, or a link to one, with no path written saying so.
    paths = [expanded, *glob.glob(expanded, root_dir=working_folder)]
    try:
        return not any(is_secret(path_in(working_folder, path)) for path in paths)
    except ToolError:
        return False


def searches_inside_files(words: list[str]) -> bool:
    """Tell whether the command reads every file under the folders it names: rg always, grep when recursive."""
    if words[0] == "rg":
        return True
    recursive_options = ("-r", "--recursive", "-d", "--directories")
    return words[0] == "grep" and any(turns_on(word, option) for word in words[1:] for option in recursive_options)


def paths_searched(words: list[str], working_folder: Path) -> list[Path]:
    """Return the paths a recursive search goes through: those named after its pattern, or else the working folder."""
    loose_words = [word for word in words[1:] if not word.startswith("-")]
    # The first loose word is taken as the pattern, also after -e: grep -r src and grep -r -e src search here.
    named_paths = loose_words[1:]
    resolved = [path_in(working_folder, os.path.expanduser(word)) for word in named_paths]
    return [path for path in resolved if path.exists()] or [working_folder]


def included_names(words: list[str]) -> list[str]:
    """Return the name patterns grep --include=... limits its search to; with none it reads every file."""
    if words[0] != "grep":
        return []
    return [value_of(word) for word in words[1:] if turns_on(word, "--include") and "=" in word]


def is_read(name: str, name_patterns: list[str]) -> bool:
    """Tell whether a recursive search reads the file of that name: every file, or only those its patterns match."""
    return not name_patterns or any(fnmatchcase(name, pattern) for pattern in name_patterns)


def may_hold_a_secret(path: Path, name_patterns: list[str]) -> bool:
    """Tell whether a secret file the search reads is under the path, or there are too many entries to look through."""
    entries_seen = 0
    for current, folder_names, file_names in os.walk(path):
        names_read = [name for name in file_names if is_read(name, name_patterns)]
        if any(is_secret(Path(current) / name) for name in names_read):
            return True
        entries_seen += len(folder_names) + len(file_names)
        if entries_seen > ENTRIES_A_SEARCH_MAY_LOOK_THROUGH:
            return True
    return False


def reads_no_secret(words: list[str], working_folder: Path) -> bool:
    """Tell whether one command of the line is a reading command, with no option it forbids, that reaches no secret."""
    if not words or words[0] not in READING_COMMANDS:
        return False
    # Right after the name, a parenthesis defines a function with that name, which a later command would run.
    if words[1:2] == ["("]:
        return False
    forbidden_options = READING_COMMANDS[words[0]]
    for word in words[1:]:
        if any(turns_on(word, option) for option in forbidden_options):
            return False
        path_written = value_of(word) if word.startswith("-") else word
        if path_written and not reaches_no_secret(path_written, working_folder):
            return False
    if searches_inside_files(words):
        name_patterns = included_names(words)
        return not any(may_hold_a_secret(path, name_patterns) for path in paths_searched(words, working_folder))
    return True


def only_reads(command: str, working_folder: Path) -> bool:
    """Tell whether the command only reads, and nothing secret, so it can run without asking."""
    # $ and ` run a command even between double quotes, { } turn into paths no word shows, and a new line
    # starts another command where the split sees a space.
    if any(character in "$`{}\n\r" for character in command):
        return False
    words = words_of(command)
    if words is None:
        return False
    words = without_harmless_redirects(words)
    if any(is_operator(word) and word not in COMMAND_SEPARATORS + GROUPING_PARENTHESES for word in words):
        return False
    return all(reads_no_secret(each, working_folder) for each in commands_in(words))


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
        """Ask unless the command only reads, and nothing secret: any other can do anything the user can."""
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
