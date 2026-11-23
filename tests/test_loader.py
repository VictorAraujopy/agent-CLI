"""Tests for the loading: the llama-server command, its start and stop, and the servers closed sessions left behind."""

import os
import socket
import stat
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from agent_cli.inference import loader
from agent_cli.inference.loader import (
    confirm_server_speaks_rpc,
    folder_of_the_servers,
    host_and_port,
    llama_server_program,
    load_model,
    record_of,
    server_command,
    stop_server,
    stop_servers_left_behind,
    wait_for_server,
)

SECONDS_A_FAKE_SERVER_RUNS = "60"
# What the real llama-server writes when the file is not a model: the cause first, then the errors it caused.
LOG_OF_A_LOAD_THAT_FAILED = (
    'echo "0.00.000.064 I srv  llama_server: initializing ..."\n'
    "echo \"0.00.054.590 E gguf_init_from_reader: invalid magic characters: 'xB', expected 'GGUF'\"\n"
    'echo "0.00.054.704 E llama_model_load: error loading model"\n'
    'echo "0.00.055.790 E srv  llama_server: exiting due to model loading error"\n'
    "exit 1"
)


@pytest.fixture(autouse=True)
def folder_of_this_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the logs and records of the servers in the test's folder, away from the ones of real sessions."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def folder_of_the_servers_in(tmp_path: Path) -> Path:
    """Return the folder where the servers of a test keep their logs and records."""
    return tmp_path / f"agent-cli-{os.getuid()}"


def program_named(folder: Path, name: str, script: str) -> Path:
    """Write a shell script with that name in the folder, ready to run, and return its path."""
    program = folder / name
    program.write_text(f"#!/bin/sh\n{script}\n")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    return program


def fake_llama_server(folder: Path, script: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a llama-server that runs the script on the PATH, in place of the real one."""
    programs = folder / "programs"
    programs.mkdir()
    program_named(programs, "llama-server", script)
    monkeypatch.setenv("PATH", os.pathsep.join([str(programs), "/usr/bin", "/bin"]))


def model_file_in(folder: Path) -> Path:
    """Write an empty .gguf file in the folder, for the fake servers, which never read it."""
    model_file = folder / "model.gguf"
    model_file.touch()
    return model_file


@pytest.fixture
def silent_address() -> Iterator[str]:
    """Listen on a port of this machine that takes connections and never says anything, and return its address."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield f"127.0.0.1:{listener.getsockname()[1]}"


@pytest.fixture
def address_that_is_not_rpc() -> Iterator[str]:
    """Answer every connection to a port of this machine with text that is not RPC, and return its address."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def answer_with_http() -> None:
        """Answer each connection with an HTTP refusal, until the listener closes."""
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            with connection:
                connection.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n" * 4)

    threading.Thread(target=answer_with_http, daemon=True).start()
    yield f"127.0.0.1:{listener.getsockname()[1]}"
    listener.close()


@pytest.fixture
def running_llama_server(tmp_path: Path) -> Iterator[subprocess.Popen]:
    """Run a process named llama-server that only waits, and end it after the test if it is still there."""
    sleeping_llama_server = tmp_path / "llama-server"
    sleeping_llama_server.symlink_to("/bin/sleep")
    process = subprocess.Popen([str(sleeping_llama_server), SECONDS_A_FAKE_SERVER_RUNS])
    # The parent of a server left behind is gone, and the system collects it as soon as it ends. Here the test is
    # its parent, so a thread collects it; otherwise it would linger after ending and seem to be still running.
    threading.Thread(target=process.wait, daemon=True).start()
    yield process
    process.kill()
    process.wait()


def ended_session_id() -> int:
    """Return the number of a process that already ended, as the session that started a server left behind."""
    session = subprocess.Popen(["true"])
    session.wait()
    return session.pid


def test_missing_llama_server_says_what_to_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no llama-server on the PATH, the error says to install llama.cpp."""
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="install llama.cpp"):
        llama_server_program()


def test_command_serves_the_model_alone_with_every_layer_on_the_devices() -> None:
    """One slot, every layer on the GPU, flash attention, the context fitted to the free memory, tools, no web page."""
    command = server_command("/bin/llama-server", Path("model.gguf"), (), 8080)
    assert command == [
        "/bin/llama-server",
        "--model", "model.gguf",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--parallel", "1",
        "--n-gpu-layers", "all",
        "--flash-attn", "on",
        "--fit", "on",
        "--fit-target", "512",
        "--fit-ctx", "4096",
        "--batch-size", "512",
        "--cache-ram", "0",
        "--jinja",
        "--log-timestamps",
        "--log-prefix",
        "--no-webui",
    ]


def test_command_of_a_split_model_names_the_rpc_servers_and_reads_the_file_into_memory() -> None:
    """A split model goes to every RPC server, and the file is read instead of mapped."""
    command = server_command("/bin/llama-server", Path("model.gguf"), ("10.0.0.2:50052", "10.0.0.3:50052"), 8080)
    assert command[-4:] == ["--rpc", "10.0.0.2:50052,10.0.0.3:50052", "--load-mode", "none"]


def test_rpc_server_address_is_split_into_host_and_port() -> None:
    """An ip:port address gives its two parts; one with no port is refused, naming the address."""
    assert host_and_port("10.0.0.2:50052") == ("10.0.0.2", 50052)
    with pytest.raises(ValueError, match="10.0.0.2"):
        host_and_port("10.0.0.2")


def test_rpc_server_that_does_not_answer_is_named_in_the_error() -> None:
    """An address where nothing listens fails at once, with the address in the error."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(ConnectionError, match=f"the RPC server 127.0.0.1:{closed_port} did not answer"):
        wait_for_server(f"127.0.0.1:{closed_port}")


def test_rpc_server_that_takes_the_connection_and_says_nothing_is_named_in_the_error(
    silent_address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real llama-server waits forever on such an address: the check gives up, saying the server may be busy."""
    monkeypatch.setattr(loader, "SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER", 1)
    with pytest.raises(ConnectionError, match=f"{silent_address} accepted the connection but did not answer"):
        confirm_server_speaks_rpc(llama_server_program(), silent_address)


def test_address_that_answers_something_else_is_not_an_rpc_server(address_that_is_not_rpc: str) -> None:
    """The real llama-server gives up on an address that answers in another protocol, and the error names it."""
    with pytest.raises(ConnectionError, match=f"{address_that_is_not_rpc} did not answer as an RPC server"):
        confirm_server_speaks_rpc(llama_server_program(), address_that_is_not_rpc)


def test_rpc_server_that_does_not_answer_stops_the_load_before_the_server_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An address where nothing listens fails the load, and llama-server never runs."""
    fake_llama_server(tmp_path, f"touch {tmp_path / 'it-ran'}", monkeypatch)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(ConnectionError, match="did not answer"):
        load_model(model_file_in(tmp_path), (f"127.0.0.1:{closed_port}",))
    assert not (tmp_path / "it-ran").exists()


def test_rpc_server_that_says_nothing_stops_the_load_before_the_server_starts(
    tmp_path: Path, silent_address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An RPC server that takes the connection and never answers fails the load, and the model is never served."""
    # The fake waits like the real one when it lists the devices, and otherwise leaves a mark that it served.
    script = f'case "$*" in *--list-devices*) exec sleep 30 ;; *) touch {tmp_path / "it-ran"}; exit 1 ;; esac'
    fake_llama_server(tmp_path, script, monkeypatch)
    monkeypatch.setattr(loader, "SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER", 1)
    with pytest.raises(ConnectionError, match="accepted the connection but did not answer"):
        load_model(model_file_in(tmp_path), (silent_address,))
    assert not (tmp_path / "it-ran").exists()


def test_missing_model_file_is_refused_before_any_server_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path with no file fails with the path, and the server never runs."""
    fake_llama_server(tmp_path, f"touch {tmp_path / 'it-ran'}", monkeypatch)
    with pytest.raises(FileNotFoundError, match="missing.gguf does not exist"):
        load_model(tmp_path / "missing.gguf")
    assert not (tmp_path / "it-ran").exists()


def test_server_that_ends_while_loading_fails_with_the_first_error_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error carries the cause, which comes before the errors it caused, and where the whole log is."""
    fake_llama_server(tmp_path, LOG_OF_A_LOAD_THAT_FAILED, monkeypatch)
    with pytest.raises(RuntimeError, match="loaded: gguf_init_from_reader: invalid magic characters") as failure:
        load_model(model_file_in(tmp_path))
    assert str(folder_of_the_servers_in(tmp_path) / f"session-{os.getpid()}.log") in str(failure.value)
    assert list(folder_of_the_servers_in(tmp_path).glob("*.session")) == []


def test_error_of_an_earlier_load_of_the_session_is_not_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log keeps every load of the session; only the lines of the load that failed now are read."""
    log_file = folder_of_the_servers() / f"session-{os.getpid()}.log"
    log_file.write_text("0.00.010.000 E llama_model_load: an error of the model loaded before\n")
    fake_llama_server(tmp_path, LOG_OF_A_LOAD_THAT_FAILED, monkeypatch)
    with pytest.raises(RuntimeError, match="invalid magic characters"):
        load_model(model_file_in(tmp_path))


def test_server_that_writes_no_error_fails_with_its_last_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no line at the error level, the last line the server wrote is the reason given."""
    fake_llama_server(tmp_path, 'echo "loading the model"\necho "out of memory"\nexit 1', monkeypatch)
    with pytest.raises(RuntimeError, match="ended before the model was loaded: out of memory"):
        load_model(model_file_in(tmp_path))


def test_real_server_given_a_file_that_is_not_a_model_says_why(tmp_path: Path) -> None:
    """The real llama-server, on a file that is not a model, fails the load with the cause it wrote."""
    not_a_model = tmp_path / "not-a-model.gguf"
    not_a_model.write_bytes(b"this is not a model" * 200)
    with pytest.raises(RuntimeError, match="invalid magic characters"):
        load_model(not_a_model)


def test_server_gets_its_key_only_through_its_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is in no argument, and none of this session's variables reach the server, the key of OpenRouter included."""
    monkeypatch.setenv("AGENT_CLI_API_KEY", "the-key-of-openrouter")
    received = tmp_path / "received"
    fake_llama_server(tmp_path, f'echo "$@" > {received}.arguments\n/usr/bin/env > {received}.environment', monkeypatch)
    with pytest.raises(RuntimeError):
        load_model(model_file_in(tmp_path))
    variables = dict(line.split("=", 1) for line in Path(f"{received}.environment").read_text().splitlines())
    # The shell that runs the fake server adds PWD, SHLVL and _ of its own.
    assert set(variables) - {"PWD", "SHLVL", "_"} == {"LLAMA_API_KEY"}
    assert len(variables["LLAMA_API_KEY"]) >= 32
    assert variables["LLAMA_API_KEY"] not in Path(f"{received}.arguments").read_text()


def test_server_of_an_ended_session_is_ended(running_llama_server: subprocess.Popen) -> None:
    """A llama-server whose session ended is ended, and its record goes with it."""
    record_of(running_llama_server.pid).write_text(str(ended_session_id()))
    stop_servers_left_behind()
    assert running_llama_server.wait(timeout=5) is not None
    assert not record_of(running_llama_server.pid).exists()


def test_server_of_an_open_session_is_left_running(running_llama_server: subprocess.Popen) -> None:
    """A llama-server whose session is still open is in use: it keeps running, and so does its record."""
    record_of(running_llama_server.pid).write_text(str(os.getpid()))
    stop_servers_left_behind()
    assert running_llama_server.poll() is None
    assert record_of(running_llama_server.pid).exists()


def test_load_ends_the_servers_of_ended_sessions_before_its_own(
    running_llama_server: subprocess.Popen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading a model first ends the server a closed session left behind, even when the new load fails."""
    record_of(running_llama_server.pid).write_text(str(ended_session_id()))
    fake_llama_server(tmp_path, "exit 1", monkeypatch)
    with pytest.raises(RuntimeError):
        load_model(model_file_in(tmp_path))
    assert running_llama_server.wait(timeout=5) is not None


def test_record_still_being_written_is_left_alone(running_llama_server: subprocess.Popen) -> None:
    """An empty record, of a session writing it right now, stops neither the cleanup nor its server."""
    record_of(running_llama_server.pid).touch()
    stop_servers_left_behind()
    assert running_llama_server.poll() is None
    assert record_of(running_llama_server.pid).exists()


def test_folder_of_the_servers_is_open_only_to_this_user(tmp_path: Path) -> None:
    """The folder is named after the user and only the user can open it, even where every user shares /tmp."""
    folder = folder_of_the_servers()
    assert folder == folder_of_the_servers_in(tmp_path)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_folder_other_users_can_write_to_is_refused(tmp_path: Path) -> None:
    """A folder of that name that another user could write to, as one made by someone else, is not used."""
    folder_of_the_servers_in(tmp_path).mkdir()
    folder_of_the_servers_in(tmp_path).chmod(0o777)
    with pytest.raises(PermissionError, match="only this user can open"):
        folder_of_the_servers()


def test_link_in_place_of_the_folder_is_refused(tmp_path: Path) -> None:
    """A link with the name of the folder, which could lead the logs anywhere, is not followed."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    folder_of_the_servers_in(tmp_path).symlink_to(elsewhere)
    with pytest.raises(PermissionError, match="only this user can open"):
        folder_of_the_servers()


def test_number_that_now_belongs_to_another_program_is_not_ended() -> None:
    """When the number of a server left behind belongs to another program now, only the record is removed."""
    other_program = subprocess.Popen(["sleep", SECONDS_A_FAKE_SERVER_RUNS])
    record_of(other_program.pid).write_text(str(ended_session_id()))
    stop_servers_left_behind()
    assert other_program.poll() is None
    assert not record_of(other_program.pid).exists()
    other_program.kill()
    other_program.wait()


def test_real_server_answers_only_with_its_key_and_ends_when_stopped(small_model_file: Path) -> None:
    """The small model is served on this machine to whoever has the key; stopping ends it and removes its record."""
    server = load_model(small_model_file)
    try:
        assert server.address.startswith("http://127.0.0.1:")
        assert record_of(server.process.pid).read_text() == str(os.getpid())
        with_the_key = {"Authorization": f"Bearer {server.key}"}
        assert httpx.get(f"{server.address}/v1/models", headers=with_the_key).status_code == 200
        assert httpx.get(f"{server.address}/v1/models").status_code == 401
    finally:
        stop_server(server)
    assert server.process.poll() is not None
    assert not record_of(server.process.pid).exists()
