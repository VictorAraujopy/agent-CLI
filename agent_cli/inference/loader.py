"""Loading: opens the .gguf file with llama.cpp, alone or split with RPC servers, and measures the memory it takes."""

import contextlib
import ctypes
import io
import socket
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import llama_cpp
from llama_cpp import Llama

PROBE_CONTEXT = 1024
SMALLER_EXTRA_CONTEXT = 2048
LARGER_EXTRA_CONTEXT = 4096
SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER = 15


@dataclass(frozen=True)
class DeviceMemory:
    """One device that holds part of the model's context: what it had free and what the model takes, in bytes."""

    name: str
    free_before_the_model: int
    model_weights: int
    any_context: int
    each_context_token: float


@dataclass(frozen=True)
class MemoryUse:
    """What a model takes of the memory of each device, and the context it was trained with, in tokens."""

    trained_context: int
    devices: tuple[DeviceMemory, ...]


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
        with socket.create_connection(address, timeout=5):
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
    library.ggml_backend_dev_name.restype = ctypes.c_char_p
    library.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    bytes_free_and_bytes_in_all = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    library.ggml_backend_dev_memory.restype = None
    library.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p, *bytes_free_and_bytes_in_all]
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
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from agent_cli.inference.loader import register_server; register_server(sys.argv[2])"
    )
    folder_of_this_package = Path(__file__).parent.parent.parent
    # -P keeps the folder the program was called from out of the places Python imports from.
    return [sys.executable, "-P", "-c", code, str(folder_of_this_package), server]


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


def open_model(model_path: Path, rpc_servers: Sequence[str], context_tokens: int) -> Llama:
    """Create the llama.cpp model with every layer on the devices and a context of the given size."""
    # Mapping the file makes this machine reserve all of it, including the layers that run on the
    # RPC servers. A model that is split is read into memory instead, so each machine holds its part.
    map_the_file_into_memory = not rpc_servers
    return open_without_touching_the_terminal(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=context_tokens,
        use_mmap=map_the_file_into_memory,
        # Without flash attention llama.cpp keeps a table of attention scores that grows with the context:
        # a long context takes far more memory, and no longer the same amount for each token.
        flash_attn=True,
    )


def free_memory_by_device() -> dict[str, int]:
    """Return how many bytes each device of llama.cpp has free, by the device's name and description."""
    library = llama_cpp_library()
    free_by_device = {}
    for index in range(library.ggml_backend_dev_count()):
        device = library.ggml_backend_dev_get(index)
        free_bytes, bytes_in_all = ctypes.c_size_t(), ctypes.c_size_t()
        library.ggml_backend_dev_memory(device, ctypes.byref(free_bytes), ctypes.byref(bytes_in_all))
        name = library.ggml_backend_dev_name(device).decode()
        description = library.ggml_backend_dev_description(device).decode()
        # A device holding more than it should reports a count that wrapped around: read as signed, it is negative.
        free_by_device[f"{name} ({description})"] = ctypes.c_int64(free_bytes.value).value
    return free_by_device


def memory_another_context_takes(model: Llama, context_tokens: int) -> dict[str, int]:
    """Create one more context of the given size on a loaded model, and return the bytes it took of each device."""
    options = llama_cpp.llama_context_params.from_buffer_copy(model.context_params)
    options.n_ctx = context_tokens
    free_before = free_memory_by_device()
    context = llama_cpp.llama_init_from_model(model.model, options)
    if not context:
        raise RuntimeError(f"llama.cpp could not create a context of {context_tokens} tokens to measure the model")
    free_with_the_context = free_memory_by_device()
    llama_cpp.llama_free(context)
    return {device: free_before[device] - free_with_the_context[device] for device in free_before}


def device_memory(
    name: str, free_before_the_model: int, free_with_the_probe: int, smaller_context: int, larger_context: int
) -> DeviceMemory:
    """Turn the measurements of one device into what the model's weights, any context and each token take of it."""
    # Two contexts of different sizes tell apart what each token takes from what a context takes whatever its size.
    each_context_token = (larger_context - smaller_context) / (LARGER_EXTRA_CONTEXT - SMALLER_EXTRA_CONTEXT)
    any_context = round(smaller_context - each_context_token * SMALLER_EXTRA_CONTEXT)
    taken_by_the_probe = free_before_the_model - free_with_the_probe
    model_weights = round(taken_by_the_probe - any_context - each_context_token * PROBE_CONTEXT)
    return DeviceMemory(name, free_before_the_model, model_weights, any_context, each_context_token)


def measure_memory_use(model_path: Path, rpc_servers: Sequence[str] = ()) -> MemoryUse:
    """Load the model with a small context and return what it takes of each device that holds part of its context."""
    silence_llama_cpp()
    connect_rpc_servers(rpc_servers)
    free_before_the_model = free_memory_by_device()
    probe = open_model(model_path, rpc_servers, PROBE_CONTEXT)
    try:
        free_with_the_probe = free_memory_by_device()
        smaller_context = memory_another_context_takes(probe, SMALLER_EXTRA_CONTEXT)
        larger_context = memory_another_context_takes(probe, LARGER_EXTRA_CONTEXT)
        trained_context = llama_cpp.llama_model_n_ctx_train(probe.model)
    finally:
        probe.close()
    devices = []
    for name in free_before_the_model:
        device = device_memory(
            name, free_before_the_model[name], free_with_the_probe[name], smaller_context[name], larger_context[name]
        )
        # A device the context takes nothing of does not limit it: the processor, when every layer is on a GPU.
        if device.each_context_token > 0:
            devices.append(device)
    return MemoryUse(trained_context, tuple(devices))


def confirm_devices_hold(model: Llama) -> None:
    """Fail, closing the model, when it takes more memory than one of the devices has."""
    for device, free_bytes in free_memory_by_device().items():
        if free_bytes < 0:
            model.close()
            raise RuntimeError(f"the model takes more memory than {device} has: its context has to be smaller")


def load_model(model_path: Path, rpc_servers: Sequence[str] = (), context_tokens: int = 0) -> Llama:
    """Open the .gguf file and return the model, ready to generate, with the given context, or the trained one if 0."""
    silence_llama_cpp()
    connect_rpc_servers(rpc_servers)
    model = open_model(model_path, rpc_servers, context_tokens)
    confirm_devices_hold(model)
    return model
