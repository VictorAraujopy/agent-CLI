"""What every model offers the conversation, local or through an API, and the error for one that does not fit."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

THINKING_OPENS = "<think>"
THINKING_CLOSES = "</think>"


class ContextFullError(Exception):
    """The conversation sent to a model takes more tokens than the model's context holds."""


class Model(ABC):
    """A model the conversation talks to, without knowing whether it runs here or through an API."""

    @abstractmethod
    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the reply, thinking first, piece by piece; ContextFullError before any piece if it does not fit."""

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
