"""What every model offers the conversation, local or through an API, and the error for one that does not fit."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

THINKING_OPENS = "<think>"
THINKING_CLOSES = "</think>"

# One message of a conversation in the chat completions format: a role, a content and, in a reply that
# asks for tools, its tool calls.
Message = dict[str, Any]


class ContextFullError(Exception):
    """The conversation sent to a model takes more tokens than the model's context holds."""


@dataclass(frozen=True)
class ToolCall:
    """A tool a reply asked to run: the id that pairs it with its result, the tool's name and its arguments as JSON."""

    id: str
    name: str
    arguments: str


def thinking_and_answer(reply: str) -> tuple[str, str]:
    """Split a reply into what the model thought, between the tags that start it, and what it answered."""
    start = reply.lstrip()
    if not start.startswith(THINKING_OPENS):
        return "", reply
    thinking, _, answer = start.removeprefix(THINKING_OPENS).partition(THINKING_CLOSES)
    return thinking.strip(), answer.lstrip()


def without_its_thinking(message: Message) -> Message:
    """Return the message as a model reads it back: a reply of a model, without what the model thought."""
    if message["role"] != "assistant":
        return message
    _, answer = thinking_and_answer(message["content"])
    return {**message, "content": answer}


class Model(ABC):
    """A model the conversation talks to, without knowing whether it runs here or through an API."""

    @abstractmethod
    def reply(self, messages: list[Message], tools: Sequence[dict[str, Any]] = ()) -> Iterator[str]:
        """Yield the reply, thinking first, piece by piece; ContextFullError before any piece if it does not fit."""

    @abstractmethod
    def tool_calls(self) -> list[ToolCall]:
        """Return the tools the last reply asked to run, in order; none when it was stopped or cut."""

    @abstractmethod
    def stop(self) -> None:
        """Ask the reply being written to end, with no error, as soon as it can; safe to call from any thread."""

    @abstractmethod
    def reply_was_cut(self) -> bool:
        """Tell whether the last reply ended because the context was full, not because the model finished it."""

    @abstractmethod
    def tokens_in_use(self) -> int:
        """Return how many tokens the conversation took in the context when the last reply ended."""

    @abstractmethod
    def context_limit(self) -> int:
        """Return how many tokens the conversation may take: the size of the model's context."""

    @abstractmethod
    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices this model has for how much it thinks before it answers."""

    @abstractmethod
    def thinking(self) -> str:
        """Return the choice in use, one of the thinking options."""

    @abstractmethod
    def set_thinking(self, option: str) -> None:
        """Choose how much the model thinks before the replies that follow: one of its thinking options."""

    @abstractmethod
    def close(self) -> None:
        """Free what the model holds; the model is not used after this."""
