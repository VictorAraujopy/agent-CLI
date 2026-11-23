"""The tools that read and change files: read_file, write_file and edit_file."""

import threading
from pathlib import Path
from typing import Any

from agent_cli.tools.tool import Tool, ToolError, ToolResult, lines_in, path_in

PATH_PARAMETER = {"type": "string", "description": "The file, relative to the working folder or absolute."}


def text_of(path: Path) -> str:
    """Return the text of a file; ToolError when there is no file there or it is not text."""
    if not path.is_file():
        raise ToolError(f"there is no file at {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{path} is not a text file in UTF-8") from None


def part_starting_at(lines: list[str], first_index: int) -> list[str]:
    """Return the lines from the first one asked, at most 500 of them and 40,000 characters in all."""
    most_lines = 500
    most_characters = 40_000
    part: list[str] = []
    characters = 0
    for line in lines[first_index : first_index + most_lines]:
        characters += len(line)
        if characters > most_characters:
            if not part:
                # A single line longer than the limit, as in a minified file, comes cut, and says so.
                part.append(f"{line[:most_characters]}\n[this line goes on: cut at {most_characters} characters]\n")
            break
        part.append(line)
    return part


def marked_lines(mark: str, text: str) -> list[str]:
    """Return the first eight lines of the text, each after the mark, and how many more there are."""
    most_lines = 8
    lines = text.splitlines()
    marked = [f"{mark} {line}" for line in lines[:most_lines]]
    if len(lines) > most_lines:
        marked.append(f"  … {len(lines) - most_lines} more lines")
    return marked


class ReadFile(Tool):
    """Reads a text file, in parts when it is long; it asks first only for a file outside the working folder."""

    name = "read_file"
    description = (
        "Read a text file. A long file comes in parts, and the end of each part says how to read the next one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": PATH_PARAMETER,
            "start_line": {"type": "integer", "description": "The line to start at, counting from 1. Defaults to 1."},
        },
        "required": ["path"],
    }

    def summary(self, arguments: dict[str, Any]) -> str:
        """Return the path of the file."""
        return arguments["path"]

    def asks_permission(self, arguments: dict[str, Any]) -> bool:
        """Ask only for a file outside the working folder, which may hold what the user never meant to share."""
        path = path_in(self.working_folder, arguments["path"])
        return not path.is_relative_to(self.working_folder)

    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Return the part of the file that starts at the line asked, and how to read on when there is more."""
        path = path_in(self.working_folder, arguments["path"])
        start_line = arguments.get("start_line", 1)
        text = text_of(path)
        lines = text.splitlines(keepends=True)
        if not lines:
            return ToolResult("(the file is empty)", "empty file")
        if not 1 <= start_line <= len(lines):
            raise ToolError(f"start_line must be between 1 and {len(lines)}, the lines of {path}")
        part = part_starting_at(lines, start_line - 1)
        if "".join(part) == text:
            return ToolResult(text, lines_in(text))
        end_line = start_line + len(part) - 1
        where = f"lines {start_line}-{end_line} of {len(lines)}"
        how_to_read_on = f"[{where}. To read on, call read_file with start_line={end_line + 1}.]"
        if end_line == len(lines):
            how_to_read_on = f"[{where}: the end of the file.]"
        return ToolResult(f"{''.join(part)}\n{how_to_read_on}", where)


class WriteFile(Tool):
    """Creates a file, or replaces the whole content of one, after the user allows it."""

    name = "write_file"
    description = (
        "Create a file with the content given, or replace the whole content of an existing file. "
        "Folders missing on the way are created."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": PATH_PARAMETER,
            "content": {"type": "string", "description": "The whole text of the file."},
        },
        "required": ["path", "content"],
    }

    def summary(self, arguments: dict[str, Any]) -> str:
        """Return the path of the file."""
        return arguments["path"]

    def preview(self, arguments: dict[str, Any]) -> str:
        """Return the first lines of the content the file will have."""
        return "\n".join(marked_lines("+", arguments["content"]))

    def asks_permission(self, arguments: dict[str, Any]) -> bool:
        """Always ask: the file may be new, or the old content lost."""
        return True

    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Write the content to the file, creating the folders it needs, and say whether it was new."""
        path = path_in(self.working_folder, arguments["path"])
        if path.is_dir():
            raise ToolError(f"{path} is a folder")
        what_happened = "replaced" if path.exists() else "created"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        lines_written = lines_in(arguments["content"])
        return ToolResult(f"{path} was {what_happened}, with {lines_written}.", f"{what_happened}, {lines_written}")


class EditFile(Tool):
    """Replaces one piece of a file's text with another, after the user allows it."""

    name = "edit_file"
    description = (
        "Replace a piece of text in a file. old_text must match the file exactly, spaces and line breaks "
        "included, and appear in it only once: when it appears more often, add the lines around it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": PATH_PARAMETER,
            "old_text": {"type": "string", "description": "The text to replace, exactly as it is in the file."},
            "new_text": {"type": "string", "description": "The text to put in its place."},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def summary(self, arguments: dict[str, Any]) -> str:
        """Return the path of the file."""
        return arguments["path"]

    def preview(self, arguments: dict[str, Any]) -> str:
        """Return the lines that leave the file, marked -, and the lines that come in, marked +."""
        lines_out = marked_lines("-", arguments["old_text"])
        lines_in_their_place = marked_lines("+", arguments["new_text"])
        return "\n".join(lines_out + lines_in_their_place)

    def asks_permission(self, arguments: dict[str, Any]) -> bool:
        """Always ask: the file changes."""
        return True

    def run(self, arguments: dict[str, Any], stop_requested: threading.Event) -> ToolResult:
        """Replace the one place where the old text is; ToolError when it is in no place or in more than one."""
        path = path_in(self.working_folder, arguments["path"])
        old_text = arguments["old_text"]
        text = text_of(path)
        times_found = text.count(old_text) if old_text else 0
        if times_found == 0:
            raise ToolError(f"old_text was not found in {path}: read the file and copy the text exactly")
        if times_found > 1:
            raise ToolError(f"old_text appears {times_found} times in {path}: add the lines around it")
        path.write_text(text.replace(old_text, arguments["new_text"]), encoding="utf-8")
        change = f"{lines_in(old_text)} replaced by {lines_in(arguments['new_text'])}"
        return ToolResult(f"{path} was changed: {change}.", change)
