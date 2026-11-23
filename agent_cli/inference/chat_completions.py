"""Generation through a chat completions server: ChatCompletionsModel, what OpenRouter and llama-server share."""

import json
import threading
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import httpx

from agent_cli.inference.model import (
    THINKING_CLOSES,
    THINKING_OPENS,
    ContextFullError,
    Message,
    Model,
    ToolCall,
    without_its_thinking,
)


def error_in_the_body(response: httpx.Response) -> dict[str, Any]:
    """Return the error in the body of a refusal, or its HTTP status when the body carries no error object."""
    response.read()
    try:
        error = response.json()["error"]
    except (ValueError, LookupError, TypeError):
        error = None
    if isinstance(error, dict):
        return error
    return {"message": f"HTTP {response.status_code} {response.reason_phrase}"}


def connection_failure(server_name: str, error: httpx.TransportError) -> ConnectionError:
    """Return the error for a connection to the server that failed, with its reason, or its kind when it has none."""
    reason = str(error) or type(error).__name__
    return ConnectionError(f"the connection to {server_name} failed: {reason}")


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


class ChatCompletionsModel(Model):
    """A model behind a server that speaks the chat completions format: each reply is one request, as a stream."""

    # The name the errors give the server.
    server_name: str
    # The field of a chunk's delta that carries the thinking, which each server names its own way.
    thinking_field: str

    def __init__(self, client: httpx.Client, context_tokens: int) -> None:
        """Keep the client that reaches the server and the size of the model's context, with no reply written yet."""
        self.client = client
        self.context_tokens = context_tokens
        self.finish_reason: str | None = None
        self.tokens_counted = 0
        self.stop_requested = threading.Event()
        # The tool calls of the reply being read, by their index in it, as their parts arrive.
        self.tool_call_parts: dict[int, dict[str, str]] = {}

    @abstractmethod
    def fields_of_this_server(self) -> dict[str, Any]:
        """Return what this server needs in a request besides the conversation, such as the model or the thinking."""

    @abstractmethod
    def failure_from(self, error: dict[str, Any]) -> Exception:
        """Return the exception for an error the server reported: ContextFullError when the conversation does not fit."""

    @abstractmethod
    def tokens_counted_in(self, chunk: dict[str, Any]) -> int | None:
        """Return the tokens in the context the server counted in this chunk, or None when it carries no count."""

    def reply(self, messages: list[Message], tools: Sequence[dict[str, Any]] = ()) -> Iterator[str]:
        """Yield the reply to the conversation one piece of text at a time, as the server streams it."""
        self.finish_reason = None
        self.tool_call_parts = {}
        self.stop_requested.clear()
        wrote_a_piece = False
        request = self.request_for(messages, tools)
        try:
            with self.client.stream("POST", "/chat/completions", json=request) as response:
                if response.status_code != 200:
                    raise self.failure_from(error_in_the_body(response))
                lines = self.lines_until_stopped(response)
                chunks = self.chunks_with_no_error(chunks_in(lines))
                for piece in self.text_pieces(chunks):
                    wrote_a_piece = True
                    yield piece
        except ContextFullError:
            if not wrote_a_piece:
                raise
            # The context filled after the reply started: the reply ends here, cut, with no error.
            self.finish_reason = "length"
        except httpx.TransportError as error:
            raise connection_failure(self.server_name, error) from error

    def request_for(self, messages: list[Message], tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Return the body of the request that asks the server for the reply to the conversation."""
        request = {
            "messages": [without_its_thinking(message) for message in messages],
            "stream": True,
            **self.fields_of_this_server(),
        }
        # A server may refuse an empty list of tools, so a reply offered none sends no list.
        if tools:
            request["tools"] = list(tools)
        return request

    def chunks_with_no_error(self, chunks: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield the chunks of a reply, and raise the error the server reports in the place of one of them."""
        for chunk in chunks:
            error = chunk.get("error")
            if error is not None:
                raise self.failure_from(error)
            yield chunk

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
            # A chunk with no choice carries only the count of tokens, or how far the server has read.
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"]
            self.keep_tool_call_parts(delta)
            thinking = delta.get(self.thinking_field)
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
        """Keep why the reply ended and the tokens the server counted, when the chunk carries them."""
        choices = chunk.get("choices")
        if choices and choices[0]["finish_reason"] is not None:
            self.finish_reason = choices[0]["finish_reason"]
        tokens = self.tokens_counted_in(chunk)
        if tokens is not None:
            self.tokens_counted = tokens

    def keep_tool_call_parts(self, delta: dict[str, Any]) -> None:
        """Join the parts of each tool call in the delta: its id and name come once, its arguments in pieces."""
        for part in delta.get("tool_calls") or []:
            call = self.tool_call_parts.setdefault(part.get("index", 0), {"id": "", "name": "", "arguments": ""})
            function = part.get("function") or {}
            call["id"] = part.get("id") or call["id"]
            call["name"] = function.get("name") or call["name"]
            call["arguments"] += function.get("arguments") or ""

    def tool_calls(self) -> list[ToolCall]:
        """Return the tools the last reply asked to run, in order; none when it was stopped or cut."""
        # A reply stopped or cut may end in the middle of the arguments of a call, which would run half written.
        if self.finish_reason is None or self.finish_reason == "length":
            return []
        calls_in_order = [self.tool_call_parts[index] for index in sorted(self.tool_call_parts)]
        return [ToolCall(call["id"], call["name"], call["arguments"]) for call in calls_in_order]

    def stop(self) -> None:
        """Ask the reply being written to end, with no error, as soon as it can; safe to call from any thread."""
        self.stop_requested.set()

    def reply_was_cut(self) -> bool:
        """Tell whether the last reply ended because it reached a limit of tokens, not because the model finished it."""
        return self.finish_reason == "length"

    def tokens_in_use(self) -> int:
        """Return the last count the server gave: the conversation sent and the reply, as far as it had counted."""
        return self.tokens_counted

    def context_limit(self) -> int:
        """Return how many tokens the conversation may take: the size of the model's context."""
        return self.context_tokens
