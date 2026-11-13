"""Generation through an API: ApiModel, the Model that sends the conversation to a model on OpenRouter."""

import json
import threading
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from agent_cli.inference.model import THINKING_CLOSES, THINKING_OPENS, ContextFullError, Model


def open_client(api_key: str) -> httpx.Client:
    """Return the HTTP client that reaches OpenRouter, sending the key of the account with every request."""
    return httpx.Client(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10),
    )


def connection_failure(error: httpx.TransportError) -> ConnectionError:
    """Return the error for a connection to OpenRouter that failed, with its reason, or its kind when it has none."""
    reason = str(error) or type(error).__name__
    return ConnectionError(f"the connection to OpenRouter failed: {reason}")


def error_in_the_body(response: httpx.Response) -> dict[str, Any]:
    """Return the error in the body of a refusal, or its HTTP status when the body is not in OpenRouter's format."""
    response.read()
    try:
        error = response.json()["error"]
    except (ValueError, LookupError, TypeError):
        error = None
    if isinstance(error, dict):
        return error
    return {"message": f"HTTP {response.status_code} {response.reason_phrase}"}


def explanation_in(error: dict[str, Any]) -> str:
    """Return what an error OpenRouter reported says: its message, and the words of the provider when it has them."""
    message = error.get("message", "no reason given")
    metadata = error.get("metadata") or {}
    said_by_the_provider = metadata.get("raw")
    if said_by_the_provider is None:
        return message
    return f"{message}: {said_by_the_provider}"


def says_context_is_full(error: dict[str, Any]) -> bool:
    """Tell whether an error says the conversation does not fit: by its type, or by its text when it has no type."""
    metadata = error.get("metadata") or {}
    typed_as_full = metadata.get("error_type") == "context_length_exceeded"
    described_as_full = "maximum context length" in explanation_in(error)
    return typed_as_full or described_as_full


def failure_reported(error: dict[str, Any]) -> Exception:
    """Return the exception for an error OpenRouter reported: ContextFullError when the conversation does not fit."""
    explanation = explanation_in(error)
    if says_context_is_full(error):
        return ContextFullError(explanation)
    return RuntimeError(f"OpenRouter reported an error: {explanation}")


def endpoints_that_keep_no_data(client: httpx.Client) -> list[dict[str, Any]]:
    """Ask OpenRouter for the endpoints, of every model, whose providers keep nothing of what they receive."""
    try:
        answer = client.get("/endpoints/zdr")
    except httpx.TransportError as error:
        raise connection_failure(error) from error
    if answer.status_code != 200:
        raise failure_reported(error_in_the_body(answer))
    return answer.json()["data"]


def context_tokens_of(model_name: str, endpoints: list[dict[str, Any]]) -> int:
    """Return the size of a model's context: the smallest among its endpoints, as OpenRouter picks one per request."""
    endpoints_of_the_model = [endpoint for endpoint in endpoints if endpoint["model_id"] == model_name]
    sizes = [endpoint["context_length"] for endpoint in endpoints_of_the_model if endpoint["context_length"]]
    if not sizes:
        raise ValueError(
            f"no provider on OpenRouter serves {model_name} without keeping data: check the name of the model"
        )
    return min(sizes)


def open_api_model(client: httpx.Client, model_name: str) -> "ApiModel":
    """Return the model of that name on OpenRouter, with the context size OpenRouter reports for it now."""
    endpoints = endpoints_that_keep_no_data(client)
    return ApiModel(client, model_name, context_tokens_of(model_name, endpoints))


def chunks_in(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield the chunk carried by each data line of a Server-Sent Events stream, up to the line that ends it."""
    for line in lines:
        field, _, value = line.partition(":")
        if field != "data":
            continue
        data = value.strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def chunks_with_no_error(chunks: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield the chunks of a reply, and raise the error OpenRouter reports in the place of one of them."""
    for chunk in chunks:
        error = chunk.get("error")
        if error is not None:
            raise failure_reported(error)
        yield chunk


def answer_of(reply: str) -> str:
    """Return a reply without the thinking that starts it; one stopped inside the thinking has no answer."""
    if not reply.startswith(THINKING_OPENS):
        return reply
    _, _, answer = reply.partition(THINKING_CLOSES)
    return answer.lstrip()


def without_its_thinking(message: dict[str, str]) -> dict[str, str]:
    """Return the message as a provider expects it back: a reply of a model, without what the model thought."""
    if message["role"] != "assistant":
        return message
    return {**message, "content": answer_of(message["content"])}


class ApiModel(Model):
    """A model that runs on the providers of OpenRouter: each reply is one request, answered as a stream."""

    def __init__(self, client: httpx.Client, model_name: str, context_tokens: int) -> None:
        """Keep the client that reaches OpenRouter, the name of the model there and the size of its context."""
        self.client = client
        self.model_name = model_name
        self.context_tokens = context_tokens
        self.thinking_effort = "max"
        self.finish_reason: str | None = None
        self.tokens_counted = 0
        self.stop_requested = threading.Event()

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the reply to the conversation one piece of text at a time, as OpenRouter streams it."""
        self.finish_reason = None
        self.stop_requested.clear()
        wrote_a_piece = False
        try:
            with self.client.stream("POST", "/chat/completions", json=self.request_for(messages)) as response:
                if response.status_code != 200:
                    raise failure_reported(error_in_the_body(response))
                lines = self.lines_until_stopped(response)
                chunks = chunks_with_no_error(chunks_in(lines))
                for piece in self.text_pieces(chunks):
                    wrote_a_piece = True
                    yield piece
        except ContextFullError:
            if not wrote_a_piece:
                raise
            # The context filled after the reply started: the reply ends here, cut, with no error.
            self.finish_reason = "length"
        except httpx.TransportError as error:
            raise connection_failure(error) from error

    def request_for(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return the body of the request that asks this model for the reply to the conversation."""
        return {
            "model": self.model_name,
            "messages": [without_its_thinking(message) for message in messages],
            "stream": True,
            "provider": {"zdr": True, "data_collection": "deny"},
            "reasoning": {"effort": self.thinking_effort},
        }

    def lines_until_stopped(self, response: httpx.Response) -> Iterator[str]:
        """Yield the lines of the response as they arrive, and end at the first one that arrives after stop()."""
        for line in response.iter_lines():
            if self.stop_requested.is_set():
                return
            yield line

    def text_pieces(self, chunks: Iterator[dict[str, Any]]) -> Iterator[str]:
        """Yield the text of each chunk: the thinking first, between its tags, and then the answer."""
        thinking_is_open = False
        for chunk in chunks:
            self.keep_how_the_reply_ended(chunk)
            delta = chunk["choices"][0]["delta"]
            thinking = delta.get("reasoning")
            answer = delta.get("content")
            if thinking and not thinking_is_open:
                thinking_is_open = True
                yield f"{THINKING_OPENS}\n"
            if thinking:
                yield thinking
            if answer and thinking_is_open:
                thinking_is_open = False
                yield f"\n{THINKING_CLOSES}\n\n"
            if answer:
                yield answer
        if thinking_is_open:
            yield f"\n{THINKING_CLOSES}\n\n"

    def keep_how_the_reply_ended(self, chunk: dict[str, Any]) -> None:
        """Keep why the reply ended and the tokens OpenRouter counted for it, when the chunk carries them."""
        finish_reason = chunk["choices"][0]["finish_reason"]
        if finish_reason is not None:
            self.finish_reason = finish_reason
        usage = chunk.get("usage")
        if usage is not None:
            self.tokens_counted = usage["total_tokens"]

    def stop(self) -> None:
        """Ask the reply being written to end, with no error, as soon as it can; safe to call from any thread."""
        self.stop_requested.set()

    def reply_was_cut(self) -> bool:
        """Tell whether the last reply ended because it reached a limit of tokens, not because the model finished it."""
        return self.finish_reason == "length"

    def tokens_in_use(self) -> int:
        """Return the tokens of the last reply that reached its end: the conversation sent and the reply."""
        return self.tokens_counted

    def context_limit(self) -> int:
        """Return how many tokens the conversation may take: the size of the model's context on OpenRouter."""
        return self.context_tokens

    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices for the thinking of a model on OpenRouter: its efforts, from none to the most."""
        return ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    def thinking(self) -> str:
        """Return the effort the model is asked to think with before the replies that follow."""
        return self.thinking_effort

    def set_thinking(self, option: str) -> None:
        """Choose the effort the model thinks with before the replies that follow: one of OpenRouter's."""
        self.thinking_effort = option

    def close(self) -> None:
        """Free nothing: the model runs on OpenRouter, and the client that reaches it serves other models too."""
