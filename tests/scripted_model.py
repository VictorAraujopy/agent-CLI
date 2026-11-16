"""A model for the tests of the screen and of the commands, whose reply is fixed and whose pieces a test can hold."""

import threading
from collections.abc import Iterator

from agent_cli.inference.model import ContextFullError, Model


class ScriptedModel(Model):
    """A model for the tests: its tokens are characters, its reply is fixed, and a test can hold each piece back."""

    def __init__(self, reply_pieces: list[str], context_limit: int = 1000, hold_pieces: bool = False) -> None:
        """Keep the pieces every reply will have, the limit, and whether each piece waits for the test to allow it."""
        self.reply_pieces = reply_pieces
        self.limit = context_limit
        self.hold_pieces = hold_pieces
        self.pieces_allowed = threading.Semaphore(0)
        self.failure_after_the_pieces: Exception | None = None
        self.cut = False
        self.reply_started = threading.Event()
        self.stop_asked = threading.Event()
        self.conversations_received: list[list[dict[str, str]]] = []
        self.tokens = 0
        self.thinking_option = "on"
        self.was_closed = False

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Refuse a conversation that passes the limit, or yield the pieces, then fail if a failure was set."""
        self.stop_asked.clear()
        self.conversations_received.append(messages)
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
