"""Loading: opens the .gguf file with llama.cpp, alone or split with RPC servers. Local model only."""

import contextlib
import ctypes
import io
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import llama_cpp
from llama_cpp import Llama

ALL_LAYERS_ON_THE_GPU = -1
CONTEXT_THE_MODEL_WAS_TRAINED_WITH = 0
SECONDS_TO_WAIT_FOR_A_SERVER = 5
SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER = 15
FOLDER_OF_THIS_PACKAGE = Path(__file__).parent.parent.parent
REGISTER_A_SERVER_IN_A_NEW_PROCESS = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from agent_cli.inference.loader import register_server; register_server(sys.argv[2])"
)


@llama_cpp.llama_log_callback
def discard_message(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:
    """Receive one message of llama.cpp and drop it, so that it is not written over the screen."""


def silence_llama_cpp() -> None:
    """Make llama.cpp hand its messages to this module instead of writing them on the terminal."""
    llama_cpp.llama_log_set(discard_message, ctypes.c_void_p(0))


def open_without_touching_the_terminal(**options: Any) -> Llama:
    """Create the llama.cpp model without the library redirecting the terminal or writing on it."""
    # With verbose off, the library points the terminal's output at nothing for the whole load, which
    # blanks a screen drawn there. With verbose on it leaves the terminal alone and prints through
    # Python instead, so those prints are caught here and verbose is turned off afterwards.
    with contextlib.redirect_stderr(io.StringIO()):
        model = Llama(**options, verbose=True)
    model.verbose = False
    return model


def host_and_port(server: str) -> tuple[str, int]:
    """Split an ip:port address into its two parts."""
    host, _, port = server.rpartition(":")
    if not host or not port.isdecimal():
        raise ValueError(f"an RPC server must be written as ip:port, but it is {server!r}")
    return host, int(port)


def wait_for_server(server: str) -> None:
    """Fail with a clear error when an RPC server does not answer within a few seconds."""
    address = host_and_port(server)
    try:
        with socket.create_connection(address, timeout=SECONDS_TO_WAIT_FOR_A_SERVER):
            pass
    except OSError as error:
        raise ConnectionError(f"the RPC server {server} did not answer: {error}") from error


def llama_cpp_library() -> ctypes.CDLL:
    """Return the llama.cpp library the bindings already use, with the RPC functions declared."""
    # The installed folder holds separate copies of the device libraries, each with its own list of
    # devices. Only the copy reached through this library is the one llama.cpp reads.
    library = llama_cpp.llama_cpp._lib
    library.ggml_backend_rpc_add_server.restype = ctypes.c_void_p
    library.ggml_backend_rpc_add_server.argtypes = [ctypes.c_char_p]
    library.ggml_backend_register.restype = None
    library.ggml_backend_register.argtypes = [ctypes.c_void_p]
    library.ggml_backend_reg_count.restype = ctypes.c_size_t
    library.ggml_backend_reg_get.restype = ctypes.c_void_p
    library.ggml_backend_reg_get.argtypes = [ctypes.c_size_t]
    library.ggml_backend_dev_count.restype = ctypes.c_size_t
    library.ggml_backend_dev_get.restype = ctypes.c_void_p
    library.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    library.ggml_backend_dev_description.restype = ctypes.c_char_p
    library.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    return library


def register_server(server: str) -> None:
    """Make one RPC server a device of llama.cpp, unless it already is one."""
    library = llama_cpp_library()
    server_devices = library.ggml_backend_rpc_add_server(server.encode())
    if server_devices is None:
        raise ConnectionError(f"the RPC server {server} refused the connection")
    registered_count = library.ggml_backend_reg_count()
    registered = [library.ggml_backend_reg_get(index) for index in range(registered_count)]
    if server_devices not in registered:
        library.ggml_backend_register(server_devices)


def device_descriptions() -> list[str]:
    """Return the description of every device llama.cpp has; an RPC server is described by its address."""
    library = llama_cpp_library()
    devices = [library.ggml_backend_dev_get(index) for index in range(library.ggml_backend_dev_count())]
    return [library.ggml_backend_dev_description(device).decode() for device in devices]


def command_that_registers(server: str) -> list[str]:
    """Return the command of a separate process that does nothing but try to register the server."""
    # -P keeps the folder the program was called from out of the places Python imports from.
    code = REGISTER_A_SERVER_IN_A_NEW_PROCESS
    return [sys.executable, "-P", "-c", code, str(FOLDER_OF_THIS_PACKAGE), server]


def confirm_server_speaks_rpc(server: str) -> None:
    """Fail with a clear error when the address is not a working RPC server."""
    # llama.cpp ends the whole process when the other side answers something that is not RPC, and
    # waits forever when it answers nothing. The first contact is made by a separate process, which
    # can die or be stopped without taking this one with it, and whose output goes nowhere.
    try:
        finished = subprocess.run(
            command_that_registers(server),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER,
        )
    except subprocess.TimeoutExpired:
        raise ConnectionError(
            f"{server} accepted the connection but did not answer as an RPC server; "
            "it may be busy with another program"
        ) from None
    if finished.returncode != 0:
        raise ConnectionError(f"{server} did not answer as an RPC server")


def connect_rpc_servers(servers: Sequence[str]) -> None:
    """Make every RPC server a device that llama.cpp splits the model across, unless it already is one."""
    if servers and not llama_cpp.llama_supports_rpc():
        raise RuntimeError(
            "llama-cpp-python was installed without RPC support: "
            "reinstall it with -DGGML_RPC=on, as the README shows"
        )
    for server in servers:
        if server in device_descriptions():
            continue
        wait_for_server(server)
        confirm_server_speaks_rpc(server)
        register_server(server)


def load_model(model_path: Path, rpc_servers: Sequence[str] = ()) -> Llama:
    """Open the .gguf file and return the model, with the context it was trained with and ready to generate."""
    silence_llama_cpp()
    connect_rpc_servers(rpc_servers)
    # Mapping the file makes this machine reserve all of it, including the layers that run on the
    # RPC servers. A model that is split is read into memory instead, so each machine holds its part.
    map_the_file_into_memory = not rpc_servers
    return open_without_touching_the_terminal(
        model_path=str(model_path),
        n_gpu_layers=ALL_LAYERS_ON_THE_GPU,
        n_ctx=CONTEXT_THE_MODEL_WAS_TRAINED_WITH,
        use_mmap=map_the_file_into_memory,
    )
