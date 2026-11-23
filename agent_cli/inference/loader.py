"""Loading: starts llama-server with the .gguf file, alone or split with RPC servers, and stops it."""

import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER = 15
SECONDS_TO_END_A_SERVER = 10


@dataclass(frozen=True)
class LocalServer:
    """A llama-server this session started: its process, the address it answers on, its key and its log."""

    process: subprocess.Popen
    address: str
    key: str = field(repr=False)
    log_file: Path
    # Where the lines of this server start in the log, which the earlier servers of the session also wrote to.
    log_start: int = 0


def llama_server_program() -> str:
    """Return the path of the llama-server program on the PATH, or fail saying what to install."""
    program = shutil.which("llama-server")
    if program is None:
        raise FileNotFoundError("llama-server was not found on the PATH: install llama.cpp, as the README shows")
    return program


def belongs_only_to_this_user(folder: Path) -> bool:
    """Tell whether the path is a folder of this user that no other user can open, and not a link to one."""
    found = folder.lstat()
    is_a_folder = stat.S_ISDIR(found.st_mode)
    owned_by_this_user = found.st_uid == os.getuid()
    closed_to_others = stat.S_IMODE(found.st_mode) & 0o077 == 0
    return is_a_folder and owned_by_this_user and closed_to_others


def folder_of_the_servers() -> Path:
    """Return this user's folder for the logs of the servers and which session started each one."""
    # On Linux every user shares the temporary folder: one another user made or can write to is not used.
    folder = Path(tempfile.gettempdir()) / f"agent-cli-{os.getuid()}"
    folder.mkdir(mode=0o700, exist_ok=True)
    if not belongs_only_to_this_user(folder):
        raise PermissionError(f"{folder} must be a folder that only this user can open")
    return folder


def record_of(server_id: int) -> Path:
    """Return the file that keeps which session started the server with that process number."""
    return folder_of_the_servers() / f"llama-server-{server_id}.session"


def is_running(process_id: int) -> bool:
    """Tell whether a process with that number exists now."""
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def program_of(process_id: int) -> str:
    """Return the name of the program a process runs, or no name when there is no such process."""
    finished = subprocess.run(["ps", "-p", str(process_id), "-o", "comm="], capture_output=True, text=True)
    return Path(finished.stdout.strip()).name


def end_process(process_id: int) -> None:
    """Ask a process that is not a child of this one to end, wait for it, and kill it if it is still there."""
    os.kill(process_id, signal.SIGTERM)
    deadline = time.monotonic() + SECONDS_TO_END_A_SERVER
    while is_running(process_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if is_running(process_id):
        os.kill(process_id, signal.SIGKILL)


def stop_servers_left_behind() -> None:
    """End the servers of sessions that closed without ending them, which keep the memory of their models taken."""
    for record in folder_of_the_servers().glob("llama-server-*.session"):
        session_id = record.read_text()
        # A session writes its record right after its server starts: one still empty is being written now.
        if not session_id.isdecimal() or is_running(int(session_id)):
            continue
        server_id = int(record.stem.removeprefix("llama-server-"))
        # The number may belong to another program by now: only a llama-server is ended.
        if program_of(server_id) == "llama-server":
            end_process(server_id)
        record.unlink()


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


def confirm_server_speaks_rpc(program: str, server: str) -> None:
    """Fail with a clear error when the address is not an RPC server that answers now."""
    # llama-server lists the devices, the RPC server's among them, and ends. An RPC server answers one
    # program at a time: one that is busy, or anything that takes the connection and says nothing, would
    # hold the load forever.
    try:
        finished = subprocess.run(
            [program, "--rpc", server, "--list-devices"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={},
            timeout=SECONDS_TO_WAIT_FOR_AN_RPC_ANSWER,
        )
    except subprocess.TimeoutExpired:
        raise ConnectionError(
            f"the RPC server {server} accepted the connection but did not answer; it may be busy with another program"
        ) from None
    if finished.returncode != 0:
        raise ConnectionError(f"{server} did not answer as an RPC server")


def check_rpc_servers(program: str, rpc_servers: Sequence[str]) -> None:
    """Fail with a clear error when an RPC server does not answer, or does not answer as one."""
    for rpc_server in rpc_servers:
        wait_for_server(rpc_server)
        confirm_server_speaks_rpc(program, rpc_server)


def free_port() -> int:
    """Return a port of this machine that nothing listens on now, chosen by the system."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def server_command(program: str, model_path: Path, rpc_servers: Sequence[str], port: int) -> list[str]:
    """Return the command that serves the model with every layer on the devices and the context fitted to them."""
    command = [
        program,
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        # A single conversation: more slots would split the context among conversations that never come.
        "--parallel", "1",
        "--n-gpu-layers", "all",
        "--flash-attn", "on",
        # The context grows up to what the devices have free, leaving 512 MiB on each, and never below 4096 tokens.
        "--fit", "on",
        "--fit-target", "512",
        "--fit-ctx", "4096",
        # The server tells how far it read after each batch, and a stop is seen only then: 512 tokens instead of
        # 2048 make the stop take under a second while a long conversation is read, at the same speed.
        "--batch-size", "512",
        # The context already keeps what was read of the conversation; a copy in memory would only take room.
        "--cache-ram", "0",
        # The chat template of the .gguf writes the tools and reads the calls back, the same way for every model.
        "--jinja",
        # Every line of the log starts with its time and its level, E for an error: a failed load is told by them.
        "--log-timestamps",
        "--log-prefix",
        "--no-webui",
    ]
    if rpc_servers:
        # Mapping the file makes this machine reserve all of it, including the layers that run on the RPC servers.
        command += ["--rpc", ",".join(rpc_servers), "--load-mode", "none"]
    return command


def size_of(file: Path) -> int:
    """Return how many bytes the file holds, or none when it does not exist yet."""
    return file.stat().st_size if file.exists() else 0


def start_server(command: list[str], key: str, log_file: Path) -> subprocess.Popen:
    """Run the command as a child process that writes to the log file and reads its key from its environment."""
    # The key goes in the environment, which other users of the machine cannot read, unlike the command.
    # Nothing else of this session's environment goes, the key of OpenRouter included.
    with open(log_file, "a") as log:
        return subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env={"LLAMA_API_KEY": key}
        )


def says_it_is_ready(client: httpx.Client) -> bool:
    """Ask the server whether the model is loaded; while it loads it answers 503, and before that, nothing."""
    try:
        return client.get("/health").status_code == 200
    except httpx.TransportError:
        return False


def error_in(line: str) -> str | None:
    """Return the message of a log line written at the error level, or None for any other line."""
    time_written, _, rest = line.partition(" ")
    level, _, message = rest.partition(" ")
    if not time_written or level != "E":
        return None
    return message.strip()


def reason_in_the_log(server: LocalServer) -> str:
    """Return the first error the server wrote, which caused the ones after it, or else its last line."""
    with open(server.log_file, "rb") as log:
        log.seek(server.log_start)
        lines = log.read().decode(errors="replace").strip().splitlines()
    errors = [error for error in map(error_in, lines) if error is not None]
    if errors:
        return errors[0]
    return lines[-1] if lines else ""


def wait_until_ready(server: LocalServer) -> None:
    """Ask the server every fifth of a second whether the model is loaded, and fail if the server ends first."""
    # The server is on this machine: a proxy set in the environment must not stand in between.
    with httpx.Client(base_url=server.address, trust_env=False) as client:
        while server.process.poll() is None:
            if says_it_is_ready(client):
                return
            time.sleep(0.2)
    raise RuntimeError(
        f"llama-server ended before the model was loaded: {reason_in_the_log(server)} "
        f"(the whole log is in {server.log_file})"
    )


def stop_server(server: LocalServer) -> None:
    """End the server and wait for it, which frees what the model takes, here and on the RPC servers."""
    server.process.terminate()
    try:
        server.process.wait(timeout=SECONDS_TO_END_A_SERVER)
    except subprocess.TimeoutExpired:
        server.process.kill()
        server.process.wait()
    record_of(server.process.pid).unlink(missing_ok=True)


def load_model(model_path: Path, rpc_servers: Sequence[str] = ()) -> LocalServer:
    """Start llama-server with the .gguf file and return it once the model is loaded and ready to answer."""
    program = llama_server_program()
    if not model_path.is_file():
        raise FileNotFoundError(f"the model file {model_path} does not exist")
    check_rpc_servers(program, rpc_servers)
    stop_servers_left_behind()
    port = free_port()
    key = secrets.token_urlsafe(32)
    log_file = folder_of_the_servers() / f"session-{os.getpid()}.log"
    log_start = size_of(log_file)
    process = start_server(server_command(program, model_path, rpc_servers, port), key, log_file)
    record_of(process.pid).write_text(str(os.getpid()))
    server = LocalServer(process, f"http://127.0.0.1:{port}", key, log_file, log_start)
    try:
        wait_until_ready(server)
    except BaseException:
        stop_server(server)
        raise
    return server
