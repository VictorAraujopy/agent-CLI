"""A model for the tests of the screen and of the commands, whose reply is fixed and whose pieces a test can hold."""

import threading
from collections.abc import Iterator, Sequence
from typing import Any

from agent_cli.inference.model import ContextFullError, Message, Model, ToolCall


class ScriptedModel(Model):
    """A model for the tests: its tokens are characters, its reply is fixed, and a test can hold each piece back."""

    def __init__(
        self,
        reply_pieces: list[str],
        context_limit: int = 1000,
        hold_pieces: bool = False,
        tool_calls_by_reply: list[list[ToolCall]] | None = None,
    ) -> None:
        """Keep the pieces every reply will have, the limit, whether each piece waits, and the tools to ask for."""
        self.reply_pieces = reply_pieces
        # The first reply that ends asks for the first list of calls, the next one for the second, then none.
        self.tool_calls_by_reply = list(tool_calls_by_reply or [])
        self.calls_of_the_last_reply: list[ToolCall] = []
        self.tools_received: list[Sequence[dict[str, Any]]] = []
        self.limit = context_limit
        self.hold_pieces = hold_pieces
        self.pieces_allowed = threading.Semaphore(0)
        self.failure_after_the_pieces: Exception | None = None
        self.cut = False
        self.reply_started = threading.Event()
        self.stop_asked = threading.Event()
        self.conversations_received: list[list[Message]] = []
        self.tokens = 0
        self.thinking_option = "on"
        self.was_closed = False

    def reply(self, messages: list[Message], tools: Sequence[dict[str, Any]] = ()) -> Iterator[str]:
        """Refuse a conversation that passes the limit, or yield the pieces, then fail if a failure was set."""
        self.stop_asked.clear()
        self.calls_of_the_last_reply = []
        self.conversations_received.append(messages)
        self.tools_received.append(tools)
        conversation_size = sum(len(message["content"]) for message in messages)
        if conversation_size > self.limit:
            raise ContextFullError(f"{conversation_size} characters do not fit in {self.limit}")
        self.tokens = conversation_size
        self.reply_started.set()
        for piece in self.reply_pieces:
            if self.hold_pieces:
                self.pieces_allowed.acquire(timeout=5)
            if self.stop_asked.is_set():
                return
            self.tokens += len(piece)
            yield piece
        if self.failure_after_the_pieces is not None:
            raise self.failure_after_the_pieces
        if self.tool_calls_by_reply:
            self.calls_of_the_last_reply = self.tool_calls_by_reply.pop(0)

    def tool_calls(self) -> list[ToolCall]:
        """Return the calls the last reply asked for: none when it was stopped before its end."""
        return self.calls_of_the_last_reply

    def allow_piece(self) -> None:
        """Let a reply that is being held write its next piece."""
        self.pieces_allowed.release()

    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices of a model whose thinking is only turned on or off."""
        return ("on", "off")

    def thinking(self) -> str:
        """Return the choice in use."""
        return self.thinking_option

    def set_thinking(self, option: str) -> None:
        """Keep the choice, for the test to look at."""
        self.thinking_option = option

    def close(self) -> None:
        """Remember that the model was closed."""
        self.was_closed = True

    def stop(self) -> None:
        """End the reply at its next piece, also when that piece is being held."""
        self.stop_asked.set()
        self.pieces_allowed.release()

    def reply_was_cut(self) -> bool:
        """Tell what the test set: whether the last reply counts as cut by a full context."""
        return self.cut

    def tokens_in_use(self) -> int:
        """Return the characters of the last conversation and of the reply written so far."""
        return self.tokens

    def context_limit(self) -> int:
        """Return the limit given when the model was made."""
        return self.limit
