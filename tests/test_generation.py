"""Tests for the generation with a local model: what LocalModel asks llama-server and how it reads the answer."""

import json
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_cli.inference.generation import LocalModel, context_tokens_of, open_local_client, open_local_model
from agent_cli.inference.loader import LocalServer, load_model, record_of
from agent_cli.inference.model import ContextFullError, ToolCall

SERVER_ADDRESS = "http://127.0.0.1:8080"
QUESTION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hello."}]
CONTEXT_OF_THE_MODEL = 4096
TOKENS_OF_THE_QUESTION = 13
TOKENS_ALREADY_READ = 5
SMALL_MODEL_TRAINED_CONTEXT = 2048
PROXY_VARIABLES = ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


def timings(read: int, written: int) -> dict[str, int]:
    """Return the count llama-server puts in a chunk: what was in its cache, what it read now, and what it wrote."""
    return {"cache_n": TOKENS_ALREADY_READ, "prompt_n": read - TOKENS_ALREADY_READ, "predicted_n": written}


def chunk_with(text: str | None, finish_reason: str | None = None, thinking: str | None = None) -> dict[str, Any]:
    """Return one streamed chunk in the shape llama-server gives, carrying an answer, a thinking or the end."""
    delta = {"content": text, "reasoning_content": thinking}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def written_chunk(text: str, written: int) -> dict[str, Any]:
    """Return a chunk with a piece of the answer and the count of the question and of what was written so far."""
    return {**chunk_with(text), "timings": timings(TOKENS_OF_THE_QUESTION, written)}


def progress_chunk(read: int) -> dict[str, Any]:
    """Return the chunk llama-server sends while it reads the conversation, before any text."""
    progress = {"total": TOKENS_OF_THE_QUESTION, "cache": TOKENS_ALREADY_READ, "processed": read, "time_ms": 10.0}
    return {**chunk_with(None), "prompt_progress": progress, "timings": timings(read, 0)}


def lines_of(*chunks: dict[str, Any]) -> list[str]:
    """Return the stream that carries the chunks, as llama-server writes it."""
    return [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]


def lines_of_a_short_reply() -> list[str]:
    """Return a whole reply: the progress of the reading, two pieces of text, and the end."""
    return lines_of(
        progress_chunk(TOKENS_ALREADY_READ),
        progress_chunk(TOKENS_OF_THE_QUESTION),
        written_chunk("Hel", 1),
        written_chunk("lo", 2),
        chunk_with(None, "stop"),
    )


@pytest.fixture
def server_process() -> Iterator[subprocess.Popen]:
    """Run a process that stands in for llama-server, and end it after the test if it is still there."""
    process = subprocess.Popen(["sleep", "60"])
    yield process
    process.kill()
    process.wait()


def model_reaching(
    handler: Callable[[httpx.Request], httpx.Response], process: subprocess.Popen, tmp_path: Path
) -> LocalModel:
    """Return a local model whose requests go to the handler, and whose server is the process given."""
    server = LocalServer(process, SERVER_ADDRESS, "the-key", tmp_path / "server.log")
    client = httpx.Client(base_url=SERVER_ADDRESS, transport=httpx.MockTransport(handler))
    return LocalModel(server, client, CONTEXT_OF_THE_MODEL)


class FakeLlamaServer:
    """Stands in for llama-server: answers every request with the same lines and remembers what it was sent."""

    def __init__(self, lines: list[str], status: int = 200) -> None:
        """Keep the lines and the status every answer carries, with no request received yet."""
        self.lines = lines
        self.status = status
        self.requests: list[httpx.Request] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        """Remember the request and answer it with the lines."""
        self.requests.append(request)
        return httpx.Response(self.status, content="".join(self.lines).encode())

    def body_received(self) -> dict[str, Any]:
        """Return the body of the last request."""
        return json.loads(self.requests[-1].content)


def refusal(message: str, error_type: str) -> list[str]:
    """Return the body of an error, as llama-server writes it."""
    return [json.dumps({"error": {"code": 400, "message": message, "type": error_type}})]


def test_request_asks_for_the_thinking_the_count_and_the_progress(
    server_process: subprocess.Popen, tmp_path: Path
) -> None:
    """One streamed request carries the conversation, the thinking switch on, and asks for the count and the progress."""
    server = FakeLlamaServer(lines_of_a_short_reply())
    list(model_reaching(server.answer, server_process, tmp_path).reply(QUESTION))
    [request] = server.requests
    assert str(request.url) == f"{SERVER_ADDRESS}/chat/completions"
    assert server.body_received() == {
        "messages": QUESTION,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "timings_per_token": True,
        "return_progress": True,
    }


def test_tool_call_is_read_as_llama_server_streams_it(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """The shape llama-server gives: no text, the id and the name with the first piece of the arguments, then pieces."""
    first_part = {"index": 0, "id": "jfO9snGT", "type": "function", "function": {"name": "read_file", "arguments": "{"}}
    pieces = ['"path":"', "notes", ".txt", '"', "}"]
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}]
    chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [first_part]}, "finish_reason": None}]})
    for piece in pieces:
        part = {"index": 0, "function": {"arguments": piece}}
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [part]}, "finish_reason": None}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    model = model_reaching(FakeLlamaServer(lines_of(*chunks)).answer, server_process, tmp_path)
    assert list(model.reply(QUESTION, [{"type": "function"}])) == []
    assert model.tool_calls() == [ToolCall("jfO9snGT", "read_file", '{"path":"notes.txt"}')]


def test_thinking_turned_off_goes_in_the_requests_that_follow(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """After set_thinking("off") the template is told the model may not think."""
    server = FakeLlamaServer(lines_of_a_short_reply())
    model = model_reaching(server.answer, server_process, tmp_path)
    model.set_thinking("off")
    list(model.reply(QUESTION))
    assert server.body_received()["chat_template_kwargs"] == {"enable_thinking": False}


def test_local_model_offers_its_thinking_on_or_off_and_starts_with_it_on(
    server_process: subprocess.Popen, tmp_path: Path
) -> None:
    """A local model has on and off as choices and starts with the thinking on."""
    model = model_reaching(FakeLlamaServer([]).answer, server_process, tmp_path)
    assert model.thinking_options() == ("on", "off")
    assert model.thinking() == "on"
    model.set_thinking("off")
    assert model.thinking() == "off"


def test_progress_gives_no_piece_and_the_reply_is_counted_with_the_question(
    server_process: subprocess.Popen, tmp_path: Path
) -> None:
    """The chunks of the reading carry no text: only the answer is yielded, and the count holds question and answer."""
    model = model_reaching(FakeLlamaServer(lines_of_a_short_reply()).answer, server_process, tmp_path)
    assert list(model.reply(QUESTION)) == ["Hel", "lo"]
    assert model.tokens_in_use() == TOKENS_OF_THE_QUESTION + 2
    assert model.reply_was_cut() is False


def test_reply_stopped_halfway_is_counted_up_to_the_last_piece_read(
    server_process: subprocess.Popen, tmp_path: Path
) -> None:
    """A stop keeps the count of the question and of the pieces read, not the count of the reply before."""
    model = model_reaching(FakeLlamaServer(lines_of_a_short_reply()).answer, server_process, tmp_path)
    reply = model.reply(QUESTION)
    assert next(reply) == "Hel"
    model.stop()
    assert list(reply) == []
    assert model.tokens_in_use() == TOKENS_OF_THE_QUESTION + 1


def test_conversation_being_read_counts_whole(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """While the server reads the conversation, the count is all of it, not only the part read so far."""
    model = model_reaching(FakeLlamaServer([]).answer, server_process, tmp_path)
    assert model.tokens_counted_in(progress_chunk(TOKENS_ALREADY_READ)) == TOKENS_OF_THE_QUESTION
    assert model.tokens_counted_in(chunk_with(None)) is None


def test_thinking_comes_from_its_own_field_between_the_tags(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """What llama-server separates as the thinking is yielded first, between the tags, and the answer after."""
    lines = lines_of(chunk_with(None, thinking="17 times 23"), chunk_with("391"), chunk_with(None, "stop"))
    model = model_reaching(FakeLlamaServer(lines).answer, server_process, tmp_path)
    assert list(model.reply(QUESTION)) == ["<think>\n", "17 times 23", "\n</think>\n\n", "391"]


def test_conversation_larger_than_the_context_does_not_fit(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """The refusal llama-server types as a context too small is a ContextFullError."""
    lines = refusal("request (4606 tokens) exceeds the available context size (4096 tokens)", "exceed_context_size_error")
    model = model_reaching(FakeLlamaServer(lines, status=400).answer, server_process, tmp_path)
    with pytest.raises(ContextFullError, match="exceeds the available context size"):
        list(model.reply(QUESTION))


def test_other_refusal_carries_what_llama_server_explained(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """Any other refusal raises an error with the message llama-server gave."""
    lines = refusal("Invalid API Key", "authentication_error")
    model = model_reaching(FakeLlamaServer(lines, status=401).answer, server_process, tmp_path)
    with pytest.raises(RuntimeError, match="llama-server reported an error: Invalid API Key"):
        list(model.reply(QUESTION))


def test_connection_that_fails_names_llama_server(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """A request that gets no answer raises a connection error that names llama-server."""

    def fail(request: httpx.Request) -> httpx.Response:
        """Stand in for a server that closed the connection."""
        raise httpx.RemoteProtocolError("server disconnected")

    with pytest.raises(ConnectionError, match="the connection to llama-server failed: server disconnected"):
        list(model_reaching(fail, server_process, tmp_path).reply(QUESTION))


def test_context_is_the_one_the_server_created() -> None:
    """The size of the context comes from the properties of the server, which fitted it to the free memory."""
    properties = {"default_generation_settings": {"n_ctx": 244736}}
    client = httpx.Client(
        base_url=SERVER_ADDRESS, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=properties))
    )
    assert context_tokens_of(client) == 244736


def test_client_reaches_the_server_with_its_key(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """The client points at the address of the server, sends its key and waits two minutes for the next line."""
    client = open_local_client(LocalServer(server_process, SERVER_ADDRESS, "the-key", tmp_path / "server.log"))
    assert str(client.base_url) == SERVER_ADDRESS
    assert client.headers["Authorization"] == "Bearer the-key"
    assert client.timeout.read == 120


def test_closing_the_model_ends_its_server(server_process: subprocess.Popen, tmp_path: Path) -> None:
    """Closing the local model ends the process of its server, which frees the memory the model takes."""
    model = model_reaching(FakeLlamaServer([]).answer, server_process, tmp_path)
    model.close()
    assert server_process.poll() is not None


class ProxyThatAnswersEverything(BaseHTTPRequestHandler):
    """Stands in for a proxy set in the environment: answers 200 to anything, and counts what reaches it."""

    requests_received = 0

    def do_GET(self) -> None:
        """Count the request and answer it with an empty object."""
        type(self).requests_received += 1
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_POST = do_GET

    def log_message(self, format: str, *arguments: Any) -> None:
        """Write nothing on the test's output."""


@pytest.fixture
def proxy_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[ProxyThatAnswersEverything]]:
    """Run the proxy and point every proxy variable of the environment at it, with no address left out of it."""
    ProxyThatAnswersEverything.requests_received = 0
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyThatAnswersEverything)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, f"http://127.0.0.1:{proxy.server_address[1]}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    yield ProxyThatAnswersEverything
    proxy.shutdown()
    proxy.server_close()


def test_requests_to_the_local_server_ignore_a_proxy_of_the_environment(
    small_model_file: Path, proxy_in_the_environment: type[ProxyThatAnswersEverything]
) -> None:
    """The load, the context and the reply go straight to the server: the proxy, which would get the key, gets nothing."""
    model = open_local_model(small_model_file)
    try:
        assert next(model.reply([{"role": "user", "content": "Once upon a time"}]))
    finally:
        model.close()
    assert proxy_in_the_environment.requests_received == 0


def test_server_is_ended_when_its_context_cannot_be_read(small_model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When asking the size of the context fails, the server just started is ended, not left holding the model."""
    servers_started: list[LocalServer] = []

    def load_and_remember(model_path: Path, rpc_servers: tuple[str, ...] = ()) -> LocalServer:
        """Load the model as usual, and remember the server."""
        server = load_model(model_path, rpc_servers)
        servers_started.append(server)
        return server

    def fail(client: httpx.Client) -> int:
        """Stand in for a server that dropped the connection when asked its properties."""
        raise ConnectionError("the connection to llama-server failed: server disconnected")

    monkeypatch.setattr("agent_cli.inference.generation.load_model", load_and_remember)
    monkeypatch.setattr("agent_cli.inference.generation.context_tokens_of", fail)
    with pytest.raises(ConnectionError):
        open_local_model(small_model_file)
    [server] = servers_started
    assert server.process.poll() is not None
    assert not record_of(server.process.pid).exists()


@pytest.fixture(scope="module")
def real_local_model(small_model_file: Path) -> Iterator[LocalModel]:
    """Serve the small model with a real llama-server for the tests below, and end the server after them."""
    model = open_local_model(small_model_file)
    yield model
    model.close()


def test_real_model_has_the_context_it_was_trained_with(real_local_model: LocalModel) -> None:
    """The small model fits whole: the server gives it the context it was trained with."""
    assert real_local_model.context_limit() == SMALL_MODEL_TRAINED_CONTEXT


def test_real_model_replies_in_pieces_and_counts_its_tokens(real_local_model: LocalModel) -> None:
    """The reply arrives as several pieces of text, and once it ends the server's count includes the conversation."""
    pieces = list(real_local_model.reply([{"role": "user", "content": "Once upon a time"}]))
    assert len(pieces) > 1
    assert real_local_model.tokens_in_use() > len(pieces)


def test_real_model_stops_when_asked_and_replies_again(real_local_model: LocalModel) -> None:
    """A stop ends the reply at once, with no error, and the next reply arrives as usual."""
    reply = real_local_model.reply([{"role": "user", "content": "Once upon a time"}])
    next(reply)
    real_local_model.stop()
    started = time.monotonic()
    list(reply)
    assert time.monotonic() - started < 1
    assert next(real_local_model.reply([{"role": "user", "content": "The cat"}]))


def test_real_reply_stopped_halfway_counts_what_it_wrote(real_local_model: LocalModel) -> None:
    """After a stop the count holds the question and what was written: the same question stopped later counts more."""
    counts = []
    for pieces_to_read in (1, 6):
        reply = real_local_model.reply([{"role": "user", "content": "Once upon a time"}])
        for _ in range(pieces_to_read):
            next(reply)
        real_local_model.stop()
        list(reply)
        counts.append(real_local_model.tokens_in_use())
    assert 0 < counts[0] < counts[1]


def test_real_conversation_larger_than_the_context_does_not_fit(real_local_model: LocalModel) -> None:
    """A conversation the context cannot hold is refused before any piece, and the model still replies afterwards."""
    too_long = [{"role": "user", "content": "the cat sat " * SMALL_MODEL_TRAINED_CONTEXT}]
    with pytest.raises(ContextFullError):
        list(real_local_model.reply(too_long))
    assert next(real_local_model.reply([{"role": "user", "content": "The dog"}]))
