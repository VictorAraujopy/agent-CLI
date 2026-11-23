"""What every tool offers: its description for the model, what the user sees of a call, and what runs it."""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """A tool could not do what the call asked; the message tells the model why, so it can try another way."""


class Ending(Enum):
    """How a tool call ended: it ran, it failed, the user refused it, or the turn stopped before it."""

    DONE = "done"
    FAILED = "failed"
    DENIED = "denied"
    NOT_RUN = "not run"


@dataclass(frozen=True)
class ToolResult:
    """What a tool call gave: the text the model reads, a short summary for the screen, and how it ended."""

    text: str
    summary: str
    ending: Ending = Ending.DONE


def path_in(working_folder: Path, path_written: str) -> Path:
    """Return the path a call names: ~ becomes the home folder, and a relative path starts at the working folder."""
    try:
        return (working_folder / Path(path_written).expanduser()).resolve()
    except (RuntimeError, OSError, ValueError) as error:
        # An unknown user after ~, a loop of links or a null character.
        raise ToolError(f"{path_written!r} is not a path that can be used: {error}") from None


def lines_in(text: str) -> str:
    """Return how many lines the text has, in words."""
    count = len(text.splitlines())
    return "1 line" if count == 1 else f"{count} lines"


class Tool(ABC):
    """A tool the model can ask for, acting from the folder the session started in."""

    # The name the model calls the tool by, what the model reads about it, and the JSON Schema of its arguments.
    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, working_folder: Path) -> None:
        """Keep the folder that relative paths start at, with its links followed, as every path is compared to it."""
        self.working_folder = working_folder.resolve()

    def schema(self) -> dict[str, Any]:
        """Return the tool as it is offered to the model, in the chat completions format."""
        function = {"name": self.name, "description": self.description, "parameters": self.parameters}
        return {"type": "function", "function": function}

    @abstractmethod
    def summary(self, arguments: dict[str, Any]) -> str:
        """Return what the call acts on, for the screen: a path or a command."""

    def preview(self, arguments: dict[str, Any]) -> str:
        """Return what the call would change, line by line, marked + or -; empty for a call that changes nothing."""
        return ""

    @abstractmethod
    def asks_permission(self, arguments: dict[str, Any]) -> bool:
        """Tell whether the user has to allow the call before it runs."""

    @abstractmethod
    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Do what the call asks and return its result; ToolError when it cannot. A long run ends at a stop."""
