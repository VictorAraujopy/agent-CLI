"""Tests for reading the system prompt."""

import re
from pathlib import Path

import pytest

from agent_cli.agent.prompt import DEFAULT_SYSTEM_PROMPT_FILE, read_system_prompt


def test_configured_file_gives_its_text(tmp_path: Path) -> None:
    """The text of the configured file is the system prompt, without the blank lines around it."""
    configured_file = tmp_path / "system_prompt.md"
    configured_file.write_text("\nAnswer in one line.\n\n", encoding="utf-8")
    assert read_system_prompt(configured_file) == "Answer in one line."


def test_default_file_is_used_when_none_is_configured() -> None:
    """With no file configured, the system prompt is the text of the project's default file."""
    default_text = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    assert read_system_prompt(None) == default_text


@pytest.mark.parametrize("content", ["", "  \n\n  "])
def test_empty_file_is_an_error_that_names_the_file(tmp_path: Path, content: str) -> None:
    """A configured file with no text, or with only blank space, stops with an error naming the file."""
    configured_file = tmp_path / "system_prompt.md"
    configured_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(configured_file))):
        read_system_prompt(configured_file)
