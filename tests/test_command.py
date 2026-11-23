"""Tests for run_command: where the command runs, what it returns, how it is ended early, and when it asks."""

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


def asks(folder: Path, shell_command: str) -> bool:
    """Tell whether run_command, from the folder, waits for the user before running the command."""
    return RunCommand(folder).asks_permission({"command": shell_command})


@pytest.mark.parametrize(
    "shell_command",
    [
        "pwd",
        "ls -la src",
        "cat notes.txt",
        "head -n 20 notes.txt | wc -l",
        "du -sh src",
        "pwd && ls -la",
        "ls src; cat notes.txt",
        "cat notes.txt || ls",
        "grep -rn 'TODO; later' . | head -5",
        "rg -n main --glob '*.py'",
        "find . -name '*.py' -type f 2>/dev/null",
        "find . -name '*.py' &>/dev/null",
        "ls -la src 2>&1",
        "find . -maxdepth 2 \\( -name '*.pdf' -o -name '*.docx' \\) -type f",
        'grep -rn "def main(" .',
        "tail -n 5 src/*.py",
        "cat /etc/hosts",
        "cat ../notes.txt",
        "ls /",
        "find / -type f \\( -iname '*cv*' -o -iname '*resume*' \\) 2>/dev/null",
        "find -L ~ -name '*.pdf'",
        "mdfind -name cv",
        "du -sh ..",
        "grep --file=/etc/hosts word notes.txt",
    ],
)
def test_command_that_only_reads_runs_without_asking(tmp_path: Path, shell_command: str) -> None:
    """Looking for and reading files, in the folder or anywhere else, alone or joined to others, needs no answer."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print()\n")
    assert not asks(tmp_path, shell_command)


@pytest.mark.parametrize(
    "shell_command",
    [
        "rm notes.txt",
        "echo hi",
        "ls; rm notes.txt",
        "ls && rm notes.txt",
        "ls || rm notes.txt",
        "ls & rm notes.txt",
        "ls > files.txt",
        "ls >&files.txt",
        "ls 2>&1 > files.txt",
        "cat notes.txt | sh",
        "ls |",
        "ls &&",
        "ls ;; cat notes.txt",
        "(rm notes.txt)",
        "ls && (rm notes.txt)",
        "ls ( ) ( rm notes.txt ) ; ls",
        "ls () ( rm notes.txt ) ; ls",
        "find . -name '*.py' -exec grep -l Catalog {} \\;",
        "ls $(rm notes.txt)",
        'grep "$(rm notes.txt)" .',
        "ls `rm notes.txt`",
        "ls\nrm notes.txt",
        "ls notes#;rm notes.txt",
        "cat 'notes.txt",
        "find . -delete",
        "find . -name '*.txt' -exec rm",
        "rg --pre ./script main",
        "rg --pre=./script main",
        "rg --pre=script main",
    ],
)
def test_command_that_can_change_something_or_run_a_program_asks(tmp_path: Path, shell_command: str) -> None:
    """Any other command, a second command after an operator, a substitution or an option that runs a program ask."""
    assert asks(tmp_path, shell_command)


@pytest.mark.parametrize(
    "shell_command",
    [
        "cat ~/.ssh/id_rsa",
        "ls ~/.ssh",
        "cat .env",
        "head config/.env.local",
        "cat ~/.aws/credentials",
        "cat ~/.netrc",
        "cat ~/.gnupg/pubring.kbx",
        "cat ~/.agent-cli/history.db",
        "grep KEY .env",
        "find . -name .env",
        "grep --file=~/.ssh/id_rsa word notes.txt",
        "grep -f.env word notes.txt",
        "cat ~no-such-user-here/notes.txt",
        "cat $HOME/.ssh/id_rsa",
        "cat {notes.txt,.env}",
        "grep -R word .",
        "grep -rS word .",
        "grep --dereference word .",
        "rg -L word",
    ],
)
def test_command_that_reads_a_secret_or_follows_links_asks(tmp_path: Path, shell_command: str) -> None:
    """A key, a .env or another secret place asks, and so do the options that follow links where no check looks."""
    assert asks(tmp_path, shell_command)


def test_link_to_a_secret_asks_when_named_or_matched_by_a_wildcard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link to a secret asks, also when only a wildcard reaches it, after ~ too; a link to anything else reads."""
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = tmp_path / "project"
    folder.mkdir()
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_rsa").write_text("key\n")
    (folder / "notes.txt").symlink_to(tmp_path / ".ssh" / "id_rsa")
    assert asks(folder, "cat notes.txt")
    assert asks(folder, "cat *")
    assert asks(folder, "cat no?es.txt")
    assert asks(folder, "cat ~/project/*")
    (folder / "notes.txt").unlink()
    (tmp_path / "plain.txt").write_text("hi\n")
    (folder / "notes.txt").symlink_to(tmp_path / "plain.txt")
    assert not asks(folder, "cat *")


def test_search_inside_the_files_of_a_folder_with_a_secret_asks(tmp_path: Path) -> None:
    """grep -r and rg read every file under a folder: one that holds a secret, at any depth, asks."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("KEY=1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print()\n")
    assert asks(tmp_path, "grep -rn KEY .")
    assert asks(tmp_path, "grep -r KEY")
    assert asks(tmp_path, "grep --recursive KEY .")
    assert asks(tmp_path, "grep -d recurse KEY .")
    assert asks(tmp_path, "rg KEY")
    assert asks(tmp_path, "grep -rn KEY src config")
    assert asks(tmp_path, "grep -r src")
    assert asks(tmp_path, "grep -r -e src")
    assert not asks(tmp_path, "grep -rn KEY src")
    assert not asks(tmp_path, "grep -r -i --include='*.py' KEY .")
    assert asks(tmp_path, "grep -r --include='*.py' --include='.env*' KEY .")
    assert asks(tmp_path, "rg --include='*.py' KEY")
    assert asks(tmp_path, "grep -r --include '*.py' KEY .")
    assert not asks(tmp_path, "grep -rn -e KEY src")
    assert not asks(tmp_path, "rg KEY src")
    assert not asks(tmp_path, "grep KEY src/main.py")
    assert not asks(tmp_path, "grep -rn KEY src/main.py")


def test_search_inside_a_folder_too_big_to_look_through_asks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A folder with more entries than the look for secrets goes through asks, since a secret could be past them."""
    monkeypatch.setattr(command, "ENTRIES_A_SEARCH_MAY_LOOK_THROUGH", 2)
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("text\n")
    assert asks(tmp_path, "grep -rn text .")
    monkeypatch.setattr(command, "ENTRIES_A_SEARCH_MAY_LOOK_THROUGH", 3)
    assert not asks(tmp_path, "grep -rn text .")
    for name in ("d", "e", "f", "g"):
        (tmp_path / "folders" / name).mkdir(parents=True)
    assert asks(tmp_path, "grep -rn text folders")


def test_summary_is_the_command(tmp_path: Path) -> None:
    """The screen shows the command as the model wrote it."""
    assert RunCommand(tmp_path).summary({"command": "ls -la"}) == "ls -la"
