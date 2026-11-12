"""The system prompt: the text of the file the user configured, or of the project's default file."""

from pathlib import Path

DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).parent / "default_system_prompt.md"


def system_prompt_file(configured_file: Path | None) -> Path:
    """Return the file the system prompt comes from: the configured one, or the default when none is."""
    if configured_file is None:
        return DEFAULT_SYSTEM_PROMPT_FILE
    return configured_file


def read_system_prompt(configured_file: Path | None) -> str:
    """Return the text of the system prompt, failing when its file is empty."""
    prompt_file = system_prompt_file(configured_file)
    # UTF-8 on purpose: without it Python uses the system's encoding, which on Windows breaks accents.
    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"the system prompt file is empty: {prompt_file}")
    return text
