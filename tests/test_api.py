"""Tests for the generation through OpenRouter. No test reaches the network: a stand-in answers the requests."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from agent_cli.inference.api import (
    ApiModel,
    answer_of,
    chunks_in,
    context_tokens_of,
    endpoints_that_keep_no_data,
    open_api_model,
    open_client,
)
from agent_cli.inference.model import ContextFullError

OPENROUTER_API = "https://openrouter.ai/api/v1"
QUESTION = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hello."}]
CONTEXT_OF_THE_MODEL = 262144
TOKENS_OF_THE_SHORT_REPLY = 14
LINES_OF_A_LONG_REPLY = 100
CONTEXT_REFUSAL_OF_A_PROVIDER = {
    "message": "The conversation is too long for the model.",
    "code": 400,
    "metadata": {"error_type": "context_length_exceeded"},
}


def chunk_with(
    text: str,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    thinking: str | None = None,
) -> dict[str, Any]:
    """Return one streamed chunk in the shape OpenRouter gives, carrying an answer, a thinking and the end."""
    delta = {"role": "assistant", "content": text, "reasoning": thinking}
    chunk: dict[str, Any] = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def chunk_with_error(error: dict[str, Any]) -> dict[str, Any]:
    """Return the chunk OpenRouter sends in the place of the next one when the reply fails halfway."""
    return {"error": error, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}


def line_with(chunk: dict[str, Any]) -> str:
    """Return the data line of a stream that carries the chunk."""
    return f"data: {json.dumps(chunk)}"


def lines_of_a_short_reply(finish_reason: str = "stop") -> list[str]:
    """Return a whole stream as OpenRouter writes it: a comment, two pieces of text, the end and the count."""
    usage = {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": TOKENS_OF_THE_SHORT_REPLY}
    return [
        ": OPENROUTER PROCESSING",
        "",
        line_with(chunk_with("Hel")),
        "",
        line_with(chunk_with("lo")),
        "",
        line_with(chunk_with("", finish_reason)),
        "",
        line_with(chunk_with("", finish_reason, usage)),
        "",
        "data: [DONE]",
        "",
    ]


def lines_of_a_long_reply() -> list[str]:
    """Return a stream with many pieces of text, which a reply stopped early never reads to the end."""
    return [line_with(chunk_with("word ")) for _ in range(LINES_OF_A_LONG_REPLY)]


class StreamOfLines(httpx.SyncByteStream):
    """The body of one answer, handed over one line at a time, that remembers how far it was read."""

    def __init__(self, lines: list[str | Exception]) -> None:
        """Keep the lines to hand over, with none read yet and the connection still open."""
        self.lines = lines
        self.lines_read = 0
        self.was_closed = False

    def __iter__(self) -> Iterator[bytes]:
        """Yield each line as the bytes of one network read; an error among the lines is the read that fails."""
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            self.lines_read += 1
            yield f"{line}\n".encode()

    def close(self) -> None:
        """Remember that the connection of this answer was closed."""
        self.was_closed = True


def model_reaching(handler: Callable[[httpx.Request], httpx.Response]) -> ApiModel:
    """Return a model whose requests go to the handler instead of the network."""
    client = httpx.Client(base_url=OPENROUTER_API, transport=httpx.MockTransport(handler))
    return ApiModel(client, "author/model", CONTEXT_OF_THE_MODEL)


def model_refused_with(status: int, body: Any) -> ApiModel:
    """Return a model whose every request is answered with the status and the JSON body."""
    return model_reaching(lambda request: httpx.Response(status, json=body))


class FakeOpenRouter:
    """Stands in for OpenRouter: answers every request with the same lines and remembers what it was sent."""

    def __init__(self, lines: list[str | Exception]) -> None:
        """Keep the lines every answer carries, with no request received yet."""
        self.lines = lines
        self.requests: list[httpx.Request] = []
        self.streams: list[StreamOfLines] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        """Remember the request and answer it with a new stream of the lines."""
        self.requests.append(request)
        self.streams.append(StreamOfLines(self.lines))
        return httpx.Response(200, stream=self.streams[-1])

    def model(self) -> ApiModel:
        """Return a model whose requests come here instead of going to the network."""
        return model_reaching(self.answer)

    def messages_received(self) -> list[dict[str, str]]:
        """Return the conversation the last request carried."""
        return json.loads(self.requests[-1].content)["messages"]


def test_reply_is_the_text_of_each_chunk_in_order() -> None:
    """The pieces of the reply are the texts of the chunks; comments, blank lines and empty texts yield nothing."""
    model = FakeOpenRouter(lines_of_a_short_reply()).model()
    assert list(model.reply(QUESTION)) == ["Hel", "lo"]


def test_request_carries_the_conversation_the_model_and_the_rules_of_every_request() -> None:
    """One streamed request goes to the chat completions, restricted to providers that keep no data, thinking on."""
    openrouter = FakeOpenRouter(lines_of_a_short_reply())
    list(openrouter.model().reply(QUESTION))
    [request] = openrouter.requests
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "author/model",
        "messages": QUESTION,
        "stream": True,
        "provider": {"zdr": True, "data_collection": "deny"},
        "reasoning": {"effort": "max"},
    }


def test_client_reaches_openrouter_with_the_key_and_the_waits() -> None:
    """The client of the real requests points at OpenRouter, sends the key and waits as long as set."""
    client = open_client("the-key")
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"
    assert client.headers["Authorization"] == "Bearer the-key"
    assert client.timeout.connect == 10
    assert client.timeout.read == 30


def test_tokens_in_use_are_the_ones_openrouter_counted() -> None:
    """Before any reply no token is counted; after one, the count is the total of the last chunk."""
    model = FakeOpenRouter(lines_of_a_short_reply()).model()
    assert model.tokens_in_use() == 0
    list(model.reply(QUESTION))
    assert model.tokens_in_use() == TOKENS_OF_THE_SHORT_REPLY


@pytest.mark.parametrize(("finish_reason", "was_cut"), [("stop", False), ("length", True)])
def test_only_a_reply_that_reached_a_limit_of_tokens_was_cut(finish_reason: str, was_cut: bool) -> None:
    """A reply the model finished was not cut; one that ended by length was."""
    model = FakeOpenRouter(lines_of_a_short_reply(finish_reason)).model()
    list(model.reply(QUESTION))
    assert model.reply_was_cut() is was_cut


def test_context_limit_is_the_size_given_to_the_model() -> None:
    """The limit of the conversation is the context size the model was created with."""
    assert FakeOpenRouter([]).model().context_limit() == CONTEXT_OF_THE_MODEL


def test_stop_ends_the_reply_and_closes_the_connection_without_reading_the_rest() -> None:
    """After stop() the reply yields nothing more, and the connection is closed with most lines unread."""
    openrouter = FakeOpenRouter(lines_of_a_long_reply())
    model = openrouter.model()
    reply = model.reply(QUESTION)
    first_piece = next(reply)
    model.stop()
    assert first_piece == "word "
    assert list(reply) == []
    assert openrouter.streams[-1].lines_read < LINES_OF_A_LONG_REPLY
    assert openrouter.streams[-1].was_closed


def test_closing_the_reply_halfway_closes_the_connection() -> None:
    """Whoever reads the reply and closes it before the end leaves no connection open."""
    openrouter = FakeOpenRouter(lines_of_a_long_reply())
    reply = openrouter.model().reply(QUESTION)
    next(reply)
    reply.close()
    assert openrouter.streams[-1].was_closed
    assert openrouter.streams[-1].lines_read < LINES_OF_A_LONG_REPLY


def test_stop_with_no_reply_being_written_is_forgotten_by_the_next_reply() -> None:
    """A stop() asked while nothing is written does not end the reply that starts afterwards."""
    model = FakeOpenRouter(lines_of_a_short_reply()).model()
    model.stop()
    assert list(model.reply(QUESTION)) == ["Hel", "lo"]


def test_stopped_reply_is_not_cut_and_keeps_the_count_of_the_reply_before() -> None:
    """A reply stopped before its last chunk forgets how the one before ended, and keeps its count of tokens."""
    model = FakeOpenRouter(lines_of_a_short_reply("length")).model()
    list(model.reply(QUESTION))
    reply = model.reply(QUESTION)
    next(reply)
    model.stop()
    list(reply)
    assert model.reply_was_cut() is False
    assert model.tokens_in_use() == TOKENS_OF_THE_SHORT_REPLY


def test_thinking_comes_first_between_its_tags_and_the_answer_after() -> None:
    """What the model thinks is yielded as it arrives, opened and closed by the tags, before what it answers."""
    lines = [
        line_with(chunk_with("", thinking="17 times 23")),
        line_with(chunk_with("", thinking=" is 391.")),
        line_with(chunk_with("391")),
        line_with(chunk_with(".", "stop")),
    ]
    pieces = list(FakeOpenRouter(lines).model().reply(QUESTION))
    assert pieces == ["<think>\n", "17 times 23", " is 391.", "\n</think>\n\n", "391", "."]


def test_reply_that_ends_while_the_model_thinks_closes_the_thinking() -> None:
    """A reply cut before any answer still closes the tag it opened."""
    lines = [line_with(chunk_with("", thinking="Let me see")), line_with(chunk_with("", "length"))]
    model = FakeOpenRouter(lines).model()
    assert list(model.reply(QUESTION)) == ["<think>\n", "Let me see", "\n</think>\n\n"]
    assert model.reply_was_cut() is True


def test_thinking_of_earlier_replies_is_not_sent_back() -> None:
    """Only the replies of the model lose their thinking; what the user or the system prompt wrote stays as it is."""
    conversation = [
        {"role": "system", "content": "<think> is a tag."},
        {"role": "user", "content": "<think>\nnot a thinking\n</think>\n\nSay hello."},
        {"role": "assistant", "content": "<think>\nA greeting.\n</think>\n\nHello."},
        {"role": "user", "content": "Again."},
    ]
    openrouter = FakeOpenRouter(lines_of_a_short_reply())
    list(openrouter.model().reply(conversation))
    assert openrouter.messages_received() == [
        conversation[0],
        conversation[1],
        {"role": "assistant", "content": "Hello."},
        conversation[3],
    ]


@pytest.mark.parametrize(
    ("reply", "answer"),
    [
        ("Hello.", "Hello."),
        ("<think>\nA greeting.\n</think>\n\nHello.", "Hello."),
        ("<think>\nstopped while thinking", ""),
        ("The <think> tag opens it and </think> closes it.", "The <think> tag opens it and </think> closes it."),
    ],
)
def test_answer_of_a_reply_is_what_comes_after_the_thinking_that_starts_it(reply: str, answer: str) -> None:
    """A reply with no thinking is all answer, one stopped inside the thinking has none, and tags later on are text."""
    assert answer_of(reply) == answer


def test_refusal_carries_what_openrouter_explained() -> None:
    """An answer that is not a success raises an error with the message of its body, never an empty reply."""
    model = model_refused_with(401, {"error": {"message": "User not found.", "code": 401}})
    with pytest.raises(RuntimeError, match="OpenRouter reported an error: User not found."):
        list(model.reply(QUESTION))


def test_refusal_of_a_provider_carries_what_the_provider_said() -> None:
    """When the message only says that a provider failed, the error also carries the words of the provider."""
    said_by_the_provider = "author/model is temporarily rate-limited upstream. Please retry shortly."
    refusal = {"message": "Provider returned error", "code": 429, "metadata": {"raw": said_by_the_provider}}
    model = model_refused_with(429, {"error": refusal})
    explanation = f"Provider returned error: {said_by_the_provider}"
    with pytest.raises(RuntimeError, match=f"OpenRouter reported an error: {explanation}"):
        list(model.reply(QUESTION))


def test_conversation_a_provider_refuses_in_its_own_words_does_not_fit() -> None:
    """A refusal with no type is still told by the words of the provider, which come apart from the message."""
    said_by_the_provider = "This model's maximum context length is 262144 tokens. Your request has 300000."
    refusal = {"message": "Provider returned error", "code": 400, "metadata": {"raw": said_by_the_provider}}
    with pytest.raises(ContextFullError, match="Your request has 300000"):
        list(model_refused_with(400, {"error": refusal}).reply(QUESTION))


@pytest.mark.parametrize("effort", ["none", "low"])
def test_thinking_chosen_for_the_model_goes_in_the_requests_that_follow(effort: str) -> None:
    """After set_thinking the requests ask for that effort, not for the one every model starts with."""
    openrouter = FakeOpenRouter(lines_of_a_short_reply())
    model = openrouter.model()
    model.set_thinking(effort)
    list(model.reply(QUESTION))
    list(model.reply(QUESTION))
    efforts_asked = [json.loads(request.content)["reasoning"] for request in openrouter.requests]
    assert efforts_asked == [{"effort": effort}, {"effort": effort}]


def test_model_on_openrouter_offers_the_efforts_and_starts_with_the_most() -> None:
    """A model on OpenRouter has the efforts of OpenRouter as choices, tells the one in use and starts at the most."""
    model = FakeOpenRouter([]).model()
    assert model.thinking_options() == ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert model.thinking() == "max"
    model.set_thinking("low")
    assert model.thinking() == "low"


def test_closed_model_leaves_the_client_working_for_the_others() -> None:
    """Closing a model on OpenRouter closes no connection: the client it shares still gets replies."""
    openrouter = FakeOpenRouter(lines_of_a_short_reply())
    closed_model = openrouter.model()
    closed_model.close()
    other_model = ApiModel(closed_model.client, "author/other-model", CONTEXT_OF_THE_MODEL)
    assert list(other_model.reply(QUESTION)) == ["Hel", "lo"]


def test_model_is_opened_with_the_context_openrouter_reports_for_it() -> None:
    """Opening a model asks OpenRouter for the endpoints and sizes the context to the ones that serve it."""
    endpoints = [{"model_id": "author/model", "context_length": 4096}]
    answer_with_the_list = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": endpoints}))
    client = httpx.Client(base_url=OPENROUTER_API, transport=answer_with_the_list)
    model = open_api_model(client, "author/model")
    assert (model.model_name, model.context_limit()) == ("author/model", 4096)
    assert model.client is client


@pytest.mark.parametrize("body", ["<html>Bad Gateway</html>", {"detail": "no error here"}, {"error": "only text"}, []])
def test_refusal_that_is_not_in_the_format_of_openrouter_is_told_by_its_status(body: Any) -> None:
    """A body that carries no error as OpenRouter writes it still raises, with the status of the answer."""
    with pytest.raises(RuntimeError, match="HTTP 502 Bad Gateway"):
        list(model_refused_with(502, body).reply(QUESTION))


def test_conversation_refused_by_openrouter_for_its_size_does_not_fit() -> None:
    """The refusal OpenRouter itself gives a conversation larger than the context has no type: its text tells."""
    message = (
        "This endpoint's maximum context length is 262144 tokens. However, you requested about 375000 "
        "tokens (375000 of text input). Please reduce the length of either one, or use the "
        "context-compression plugin to compress your prompt automatically."
    )
    refusal = {"message": message, "code": 400, "metadata": {"provider_name": None}}
    with pytest.raises(ContextFullError, match="maximum context length is 262144 tokens"):
        list(model_refused_with(400, {"error": refusal}).reply(QUESTION))


def test_conversation_refused_by_a_provider_for_its_size_does_not_fit() -> None:
    """The refusal of a provider is told by the type OpenRouter gives it, also when it arrives inside the stream."""
    with pytest.raises(ContextFullError):
        list(model_refused_with(400, {"error": CONTEXT_REFUSAL_OF_A_PROVIDER}).reply(QUESTION))
    lines = [": OPENROUTER PROCESSING", line_with(chunk_with_error(CONTEXT_REFUSAL_OF_A_PROVIDER))]
    with pytest.raises(ContextFullError):
        list(FakeOpenRouter(lines).model().reply(QUESTION))


def test_context_that_fills_after_the_reply_started_cuts_the_reply() -> None:
    """Once a piece was written the conversation did fit: the reply ends there, cut, with no error."""
    lines = [line_with(chunk_with("Once upon")), line_with(chunk_with_error(CONTEXT_REFUSAL_OF_A_PROVIDER))]
    model = FakeOpenRouter(lines).model()
    assert list(model.reply(QUESTION)) == ["Once upon"]
    assert model.reply_was_cut() is True


def test_failure_halfway_through_the_reply_raises_after_the_pieces_already_written() -> None:
    """An error sent in the place of a chunk ends the reply with what OpenRouter explained."""
    rate_limit = {"message": "Rate limit exceeded", "code": 429}
    lines = [
        line_with(chunk_with("Once upon")),
        line_with(chunk_with_error(rate_limit)),
        line_with(chunk_with("a time")),
    ]
    reply = FakeOpenRouter(lines).model().reply(QUESTION)
    assert next(reply) == "Once upon"
    with pytest.raises(RuntimeError, match="OpenRouter reported an error: Rate limit exceeded"):
        next(reply)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [(httpx.ConnectError("no route to host"), "no route to host"), (httpx.ReadTimeout(""), "ReadTimeout")],
)
def test_connection_that_fails_is_reported_as_such(failure: httpx.TransportError, reason: str) -> None:
    """A request that never gets an answer raises a connection error with the reason, or the kind when it has none."""

    def fail(request: httpx.Request) -> httpx.Response:
        """Stand in for a network that fails before any answer."""
        raise failure

    with pytest.raises(ConnectionError, match=f"the connection to OpenRouter failed: {reason}"):
        list(model_reaching(fail).reply(QUESTION))


def test_connection_that_drops_halfway_through_the_reply_is_reported_as_such() -> None:
    """A read that fails after some pieces raises a connection error, after the pieces already written."""
    lines = [line_with(chunk_with("Once upon")), httpx.ReadError("connection reset")]
    reply = FakeOpenRouter(lines).model().reply(QUESTION)
    assert next(reply) == "Once upon"
    with pytest.raises(ConnectionError, match="the connection to OpenRouter failed: connection reset"):
        next(reply)


def test_only_data_lines_carry_chunks() -> None:
    """Comments, blank lines and other fields are skipped, and a data line needs no space after the colon."""
    lines = [": comment", "", "event: message", 'data: {"n": 1}', 'data:{"n": 2}', "id: 7"]
    assert list(chunks_in(lines)) == [{"n": 1}, {"n": 2}]


def test_stream_ends_at_the_done_mark() -> None:
    """The mark that ends the stream is not read as a chunk, and nothing after it is read."""
    lines = ['data: {"n": 1}', "data: [DONE]", 'data: {"n": 2}']
    assert list(chunks_in(lines)) == [{"n": 1}]


def test_endpoints_that_keep_no_data_are_asked_of_openrouter() -> None:
    """The list comes from the route of OpenRouter for the endpoints with Zero Data Retention."""
    endpoints = [{"model_id": "author/model", "context_length": 4096}]
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        """Remember the request and answer it with the list."""
        requests.append(request)
        return httpx.Response(200, json={"data": endpoints})

    client = httpx.Client(base_url=OPENROUTER_API, transport=httpx.MockTransport(answer))
    assert endpoints_that_keep_no_data(client) == endpoints
    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", "https://openrouter.ai/api/v1/endpoints/zdr")
    ]


def test_list_of_endpoints_that_is_refused_or_unreachable_raises() -> None:
    """A refusal raises what OpenRouter explained, and a failed connection raises a connection error."""
    refusing = httpx.Client(
        base_url=OPENROUTER_API,
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"error": {"message": "Down."}})),
    )
    with pytest.raises(RuntimeError, match="OpenRouter reported an error: Down."):
        endpoints_that_keep_no_data(refusing)

    def fail(request: httpx.Request) -> httpx.Response:
        """Stand in for a network that fails before any answer."""
        raise httpx.ConnectError("no route to host")

    unreachable = httpx.Client(base_url=OPENROUTER_API, transport=httpx.MockTransport(fail))
    with pytest.raises(ConnectionError, match="the connection to OpenRouter failed: no route to host"):
        endpoints_that_keep_no_data(unreachable)


def test_context_of_a_model_is_the_smallest_among_its_endpoints() -> None:
    """Endpoints of other models and endpoints that report no context size do not count."""
    endpoints = [
        {"model_id": "author/model", "context_length": 1048576},
        {"model_id": "author/model", "context_length": 1000000},
        {"model_id": "author/model", "context_length": 0},
        {"model_id": "author/model", "context_length": None},
        {"model_id": "author/other-model", "context_length": 8192},
    ]
    assert context_tokens_of("author/model", endpoints) == 1000000


@pytest.mark.parametrize("endpoints", [[], [{"model_id": "author/model", "context_length": None}]])
def test_model_no_provider_serves_without_keeping_data_is_refused(endpoints: list[dict[str, Any]]) -> None:
    """A name that is not in the list, or that has no context size there, raises an error that names the model."""
    with pytest.raises(ValueError, match="author/model"):
        context_tokens_of("author/model", endpoints)
