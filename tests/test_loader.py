"""Tests for the model loader."""

import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import llama_cpp
import pytest
from llama_cpp import Llama

from agent_cli.inference import loader
from agent_cli.inference.loader import (
    command_that_registers,
    connect_rpc_servers,
    host_and_port,
    load_model,
    register_server,
    wait_for_server,
)

SECONDS_TO_WAIT_FOR_A_HELPER_PROCESS = 60
RPC_SERVER_OF_THIS_MACHINE = """
import ctypes
import sys

import llama_cpp

library = llama_cpp.llama_cpp._lib
llama_cpp.llama_backend_init()
library.ggml_backend_dev_by_type.restype = ctypes.c_void_p
library.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
processor = (ctypes.c_void_p * 1)(library.ggml_backend_dev_by_type(0))
library.ggml_backend_rpc_start_server.restype = None
library.ggml_backend_rpc_start_server.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
]
library.ggml_backend_rpc_start_server(sys.argv[1].encode(), None, 1, 1, processor)
"""


@pytest.fixture
def port_with_nothing_listening() -> int:
    """Return a port of this machine that was free a moment ago, so a connection to it is refused."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def answer_connections(listener: socket.socket, hang_up: bool, kept_open: list[socket.socket]) -> None:
    """Accept every connection to the listener, closing it at once or keeping it open and silent."""
    while True:
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        if hang_up:
            connection.close()
        else:
            kept_open.append(connection)


def port_that_accepts_connections(hang_up: bool) -> Iterator[int]:
    """Yield a port of this machine that accepts connections but is not an RPC server."""
    kept_open: list[socket.socket] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        threading.Thread(target=answer_connections, args=(listener, hang_up, kept_open), daemon=True).start()
        yield listener.getsockname()[1]
    for connection in kept_open:
        connection.close()


@pytest.fixture
def rpc_support() -> None:
    """Skip the test when llama-cpp-python was installed without the RPC backend."""
    if not llama_cpp.llama_supports_rpc():
        pytest.skip("llama-cpp-python was installed without RPC support")


@pytest.fixture
def rpc_server_of_this_machine(rpc_support: None, port_with_nothing_listening: int) -> Iterator[str]:
    """Start a real RPC server of llama.cpp on this machine, yield its address and stop it afterwards."""
    address = f"127.0.0.1:{port_with_nothing_listening}"
    server = subprocess.Popen(
        [sys.executable, "-c", RPC_SERVER_OF_THIS_MACHINE, address],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    give_up_at = time.monotonic() + SECONDS_TO_WAIT_FOR_A_HELPER_PROCESS
    while time.monotonic() < give_up_at:
        try:
            socket.create_connection(("127.0.0.1", port_with_nothing_listening), timeout=1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield address
    server.kill()
    server.wait()


@pytest.fixture
def port_that_hangs_up() -> Iterator[int]:
    """Yield a port that accepts a connection and closes it at once, as a wrong service would."""
    yield from port_that_accepts_connections(hang_up=True)


@pytest.fixture
def port_that_stays_silent() -> Iterator[int]:
    """Yield a port that accepts a connection and never answers, as a busy RPC server would."""
    yield from port_that_accepts_connections(hang_up=False)


def test_missing_file_is_refused(tmp_path: Path) -> None:
    """A path with no file raises an error that names the path."""
    missing_file = tmp_path / "missing.gguf"
    with pytest.raises(ValueError, match="missing.gguf"):
        load_model(missing_file)


def test_file_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    """A file that is not a GGUF raises an error instead of returning a broken model."""
    not_a_model = tmp_path / "not-a-model.gguf"
    not_a_model.write_text("this is not a model")
    with pytest.raises(ValueError, match="not-a-model.gguf"):
        load_model(not_a_model)


def test_model_is_loaded_with_the_context_it_was_trained_with(small_model_file: Path) -> None:
    """A real model loads with the context size stored in its .gguf and splits text into tokens."""
    model = load_model(small_model_file)
    assert isinstance(model, Llama)
    assert model.n_ctx() == 2048
    assert len(model.tokenize(b"hello world")) > 0


def test_llama_cpp_is_asked_for_the_context_stored_in_the_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader passes n_ctx=0, which tells llama.cpp to take the context size from the .gguf."""
    options_given_to_llama_cpp = {}
    monkeypatch.setattr(
        loader, "open_without_touching_the_terminal", options_given_to_llama_cpp.update
    )
    load_model(Path("any.gguf"))
    assert options_given_to_llama_cpp["n_ctx"] == 0


def test_loading_writes_nothing_on_the_terminal(
    small_model_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Loading stays silent, because anything written would land on top of the screen."""
    load_model(small_model_file)
    written = capfd.readouterr()
    assert written.out == ""
    assert written.err == ""


def test_generating_writes_nothing_on_the_terminal(
    small_model_file: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """After a load the library stays silent, also when it reuses the start of the previous text."""
    model = load_model(small_model_file)
    model.create_completion("Once upon a time", max_tokens=4)
    model.create_completion("Once upon a time", max_tokens=4)
    written = capfd.readouterr()
    assert written.out == ""
    assert written.err == ""


def test_loading_leaves_the_output_of_the_terminal_in_place(
    small_model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal's output is never pointed elsewhere during a load, so a screen drawn there keeps working."""
    output_before = os.fstat(2)
    output_during_the_load = []

    @llama_cpp.llama_log_callback
    def remember_the_output(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:
        """Record where the terminal's error output points each time llama.cpp says something."""
        output_during_the_load.append(os.fstat(2))

    def listen_to_llama_cpp() -> None:
        """Stand in for the silencing, so that the messages of the load reach this test."""
        llama_cpp.llama_log_set(remember_the_output, ctypes.c_void_p(0))

    monkeypatch.setattr(loader, "silence_llama_cpp", listen_to_llama_cpp)
    try:
        load_model(small_model_file)
    finally:
        llama_cpp.llama_log_set(loader.discard_message, ctypes.c_void_p(0))
    assert output_during_the_load
    assert all(
        (output.st_dev, output.st_ino) == (output_before.st_dev, output_before.st_ino)
        for output in output_during_the_load
    )


@pytest.mark.parametrize(
    ("rpc_servers", "file_is_mapped"), [((), True), (("10.0.0.2:50052",), False)]
)
def test_model_split_with_rpc_servers_is_read_instead_of_mapped(
    monkeypatch: pytest.MonkeyPatch, rpc_servers: tuple[str, ...], file_is_mapped: bool
) -> None:
    """A split model is not mapped, so this machine reserves only the layers that run on it."""
    options_given_to_llama_cpp = {}
    monkeypatch.setattr(loader, "connect_rpc_servers", lambda servers: None)
    monkeypatch.setattr(
        loader, "open_without_touching_the_terminal", options_given_to_llama_cpp.update
    )
    load_model(Path("any.gguf"), rpc_servers)
    assert options_given_to_llama_cpp["use_mmap"] is file_is_mapped


def test_address_is_split_into_host_and_port() -> None:
    """An ip:port address becomes the host as text and the port as a number."""
    assert host_and_port("10.0.0.2:50052") == ("10.0.0.2", 50052)


@pytest.mark.parametrize("wrong_address", ["10.0.0.2", ":50052", "10.0.0.2:port", ""])
def test_address_that_is_not_ip_and_port_is_refused(wrong_address: str) -> None:
    """An address without a host or without a numeric port raises an error that shows the format."""
    with pytest.raises(ValueError, match="ip:port"):
        host_and_port(wrong_address)


def test_server_that_does_not_answer_is_reported(port_with_nothing_listening: int) -> None:
    """A server that cannot be reached raises a connection error instead of hanging the load."""
    with pytest.raises(ConnectionError, match="did not answer"):
        wait_for_server(f"127.0.0.1:{port_with_nothing_listening}")


def test_server_that_stays_silent_is_given_up_on_after_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that never answers is waited for the set number of seconds and then reported."""
    waits_asked = []

    def never_connect(address: tuple[str, int], timeout: float) -> None:
        """Stand in for a connection attempt to a machine that does not answer."""
        waits_asked.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(loader.socket, "create_connection", never_connect)
    with pytest.raises(ConnectionError, match="did not answer"):
        wait_for_server("10.0.0.2:50052")
    assert waits_asked == [5]


def test_separate_process_runs_from_any_folder_and_reaches_llama_cpp(
    rpc_support: None, tmp_path: Path, port_with_nothing_listening: int
) -> None:
    """The process that makes the first contact finds this package and gets llama.cpp's own answer."""
    finished = subprocess.run(
        command_that_registers(f"127.0.0.1:{port_with_nothing_listening}"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=SECONDS_TO_WAIT_FOR_A_HELPER_PROCESS,
    )
    assert "refused the connection" in finished.stderr


def test_separate_process_ignores_modules_of_the_folder_it_runs_in(
    rpc_support: None, tmp_path: Path, port_with_nothing_listening: int
) -> None:
    """A file named like a standard module, left in the current folder, is never run by that process."""
    (tmp_path / "socket.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('was-imported').touch()\n"
    )
    subprocess.run(
        command_that_registers(f"127.0.0.1:{port_with_nothing_listening}"),
        cwd=tmp_path,
        capture_output=True,
        timeout=SECONDS_TO_WAIT_FOR_A_HELPER_PROCESS,
    )
    assert not (tmp_path / "was-imported").exists()


def test_connected_server_becomes_one_device_of_llama_cpp(rpc_server_of_this_machine: str) -> None:
    """Connecting a real RPC server makes it a device of llama.cpp, once, however often it is connected."""
    connect_twice_and_count_the_devices_of_the_server = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from agent_cli.inference import loader; loader.silence_llama_cpp(); "
        "loader.connect_rpc_servers((sys.argv[2],)); loader.connect_rpc_servers((sys.argv[2],)); "
        "print(loader.device_descriptions().count(sys.argv[2]))"
    )
    command = [
        sys.executable,
        "-P",
        "-c",
        connect_twice_and_count_the_devices_of_the_server,
        str(Path(__file__).parent.parent),
        rpc_server_of_this_machine,
    ]
    finished = subprocess.run(
        command, capture_output=True, text=True, timeout=SECONDS_TO_WAIT_FOR_A_HELPER_PROCESS
    )
    assert finished.stdout.strip() == "1"


def test_no_servers_leaves_everything_on_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no RPC server configured, connecting does nothing, even on a library without RPC support."""
    monkeypatch.setattr(llama_cpp, "llama_supports_rpc", lambda: False)
    connect_rpc_servers(())


def test_every_server_is_waited_for_confirmed_and_then_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connecting checks each server and registers it with llama.cpp, in the order they were given."""
    steps = []
    monkeypatch.setattr(llama_cpp, "llama_supports_rpc", lambda: True)
    monkeypatch.setattr(loader, "device_descriptions", lambda: [])
    monkeypatch.setattr(loader, "wait_for_server", lambda server: steps.append(("wait", server)))
    monkeypatch.setattr(
        loader, "confirm_server_speaks_rpc", lambda server: steps.append(("confirm", server))
    )
    monkeypatch.setattr(loader, "register_server", lambda server: steps.append(("register", server)))
    connect_rpc_servers(("10.0.0.2:50052", "10.0.0.3:50052"))
    assert steps == [
        ("wait", "10.0.0.2:50052"),
        ("confirm", "10.0.0.2:50052"),
        ("register", "10.0.0.2:50052"),
        ("wait", "10.0.0.3:50052"),
        ("confirm", "10.0.0.3:50052"),
        ("register", "10.0.0.3:50052"),
    ]


def test_server_that_is_already_a_device_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server registered earlier in this process is not contacted again."""
    steps = []
    monkeypatch.setattr(llama_cpp, "llama_supports_rpc", lambda: True)
    monkeypatch.setattr(loader, "device_descriptions", lambda: ["10.0.0.2:50052"])
    monkeypatch.setattr(loader, "wait_for_server", lambda server: steps.append(("wait", server)))
    monkeypatch.setattr(
        loader, "confirm_server_speaks_rpc", lambda server: steps.append(("confirm", server))
    )
    monkeypatch.setattr(loader, "register_server", lambda server: steps.append(("register", server)))
    connect_rpc_servers(("10.0.0.2:50052",))
    assert steps == []


def test_address_that_is_not_an_rpc_server_is_reported(
    rpc_support: None, port_that_hangs_up: int
) -> None:
    """An open port that is not an RPC server raises an error instead of ending the program."""
    with pytest.raises(ConnectionError, match="did not answer as an RPC server"):
        connect_rpc_servers((f"127.0.0.1:{port_that_hangs_up}",))


def test_server_that_accepts_and_never_answers_is_given_up_on(
    rpc_support: None, port_that_stays_silent: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that takes the connection and says nothing raises an error instead of waiting forever."""
    monkeypatch.setattr(loader, "SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER", 2)
    with pytest.raises(ConnectionError, match="accepted the connection but did not answer"):
        connect_rpc_servers((f"127.0.0.1:{port_that_stays_silent}",))


def test_devices_of_llama_cpp_are_described_as_text(rpc_support: None) -> None:
    """The list that tells whether a server is already a device comes from llama.cpp itself."""
    descriptions = loader.device_descriptions()
    assert descriptions
    assert all(isinstance(description, str) for description in descriptions)


def test_registering_a_server_that_is_not_there_is_reported(
    rpc_support: None, port_with_nothing_listening: int
) -> None:
    """Registering reaches llama.cpp for real, and a server it cannot connect to raises an error."""
    with pytest.raises(ConnectionError, match="refused the connection"):
        register_server(f"127.0.0.1:{port_with_nothing_listening}")


def test_servers_are_refused_when_the_library_has_no_rpc_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library compiled without RPC raises an error that says how to reinstall it."""
    monkeypatch.setattr(llama_cpp, "llama_supports_rpc", lambda: False)
    with pytest.raises(RuntimeError, match="GGML_RPC"):
        connect_rpc_servers(("10.0.0.2:50052",))
