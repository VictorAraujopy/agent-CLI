"""The tools a session offers the model, and how a call the model wrote is read, checked and run."""

import json
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_cli.tools.command import RunCommand
from agent_cli.tools.files import EditFile, ReadFile, WriteFile
from agent_cli.tools.tool import Ending, Tool, ToolError, ToolResult


@dataclass(frozen=True)
class ToolUse:
    """A call the model wrote, as the toolbox read it: what the user sees of it, and what runs it."""

    name: str
    summary: str
    preview: str
    asks_permission: bool
    # The tool and the arguments that run the call. With no tool, the call cannot run, and problem says why.
    tool: Tool | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    problem: str = ""


def arguments_in(arguments_written: str) -> dict[str, Any]:
    """Return the arguments the model wrote as a JSON object, where no text means none; ToolError otherwise."""
    if not arguments_written.strip():
        return {}
    try:
        arguments = json.loads(arguments_written)
    except json.JSONDecodeError as error:
        raise ToolError(f"the arguments are not valid JSON: {error}") from None
    if not isinstance(arguments, dict):
        raise ToolError("the arguments must be a JSON object")
    return arguments


def is_of_type(value: Any, json_type: str) -> bool:
    """Tell whether a value has the JSON type a tool's parameter names: string or integer."""
    if json_type == "integer":
        # In Python true and false are integers too, but not in JSON.
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str)


def check_arguments(arguments: dict[str, Any], parameters: dict[str, Any]) -> None:
    """Raise ToolError when an argument the tool needs is missing, or one has the wrong type."""
    for name in parameters["required"]:
        if name not in arguments:
            raise ToolError(f"the argument {name} is missing")
    for name, value in arguments.items():
        parameter = parameters["properties"].get(name)
        if parameter is not None and not is_of_type(value, parameter["type"]):
            raise ToolError(f"the argument {name} must be of the type {parameter['type']}")


def failure(reason: str) -> ToolResult:
    """Return the result of a call that failed, with the reason for the model and for the screen."""
    return ToolResult(f"error: {reason}", reason, Ending.FAILED)


class Toolbox:
    """The tools of a session: reading, writing and editing files, and running commands, from one working folder."""

    def __init__(self, working_folder: Path) -> None:
        """Make every tool act from the working folder, the one relative paths start at and commands run in."""
        tools = [ReadFile(working_folder), WriteFile(working_folder), EditFile(working_folder), RunCommand(working_folder)]
        self.tools = {tool.name: tool for tool in tools}
        # Automode approvals: set of (tool_name, arguments_hash) for calls approved to run without re-prompting
        self.automode_approvals: set[tuple[str, str]] = set()

    def schemas(self) -> list[dict[str, Any]]:
        """Return every tool as it is offered to the model."""
        return [tool.schema() for tool in self.tools.values()]

    def read(self, name: str, arguments_written: str) -> ToolUse:
        """Return the call as the toolbox reads it; a call that cannot run says why, and shows what the model wrote."""
        tool = self.tools.get(name)
        if tool is None:
            names = ", ".join(self.tools)
            return ToolUse(name, arguments_written, "", False, problem=f"there is no tool {name}: the tools are {names}")
        try:
            arguments = arguments_in(arguments_written)
            check_arguments(arguments, tool.parameters)
            asks_permission = tool.asks_permission(arguments)
        except ToolError as error:
            return ToolUse(name, arguments_written, "", False, problem=str(error))
        summary = tool.summary(arguments)
        preview = tool.preview(arguments)
        return ToolUse(name, summary, preview, asks_permission, tool, arguments)

    def run(self, use: ToolUse, stop_requested: threading.Event) -> ToolResult:
        """Run the call and return its result; a call that cannot run, or a tool that fails, gives the reason."""
        if use.tool is None:
            return failure(use.problem)
        try:
            return use.tool.run(use.arguments, stop_requested)
        except (ToolError, OSError) as error:
            return failure(str(error))

    def _hash_arguments(self, arguments: dict[str, Any]) -> str:
        """Create a consistent hash of tool arguments for automode tracking."""
        return hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()

    def is_automode_approved(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check if this tool call has been approved for automode (run without re-prompting)."""
        args_hash = self._hash_arguments(arguments)
        return (tool_name, args_hash) in self.automode_approvals

    def add_automode_approval(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Mark this tool call as approved for automode so future identical calls won't prompt."""
        args_hash = self._hash_arguments(arguments)
        self.automode_approvals.add((tool_name, args_hash))

    def clear_automode_approvals(self) -> None:
        """Clear all automode approvals."""
        self.automode_approvals.clear()

    def list_automode_approvals(self) -> list[tuple[str, str]]:
        """List all current automode approvals."""
        return list(self.automode_approvals)