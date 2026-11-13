"""The banner: the box that opens the conversation, with the butterfly, the name and the session facts."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.markup import escape
from textual.reactive import reactive
from textual.widgets import Static

from agent_cli.screen.butterfly import Butterfly
from agent_cli.screen.theme import MUTED_MARKUP


def describe_model(model_name: str | None) -> str:
    """Return the text that tells which model is in use."""
    if model_name is None:
        return "none loaded"
    return model_name


def describe_context(context_tokens: int | None) -> str:
    """Return the text that tells how many tokens the context of the model in use holds."""
    if context_tokens is None:
        return "not sized yet"
    return f"{context_tokens} tokens"


def shorten_home(directory: Path, home: Path) -> str:
    """Return the directory with the home folder written as ~, when it is inside it."""
    if directory.is_relative_to(home):
        return str(Path("~") / directory.relative_to(home))
    return str(directory)


def info_row(label: str, value: str) -> str:
    """Return one line of session facts: a muted label, padded, followed by its value."""
    padded_label = label.ljust(10)
    return f"[{MUTED_MARKUP}]{padded_label}[/] {escape(value)}"


class Banner(Horizontal):
    """The box with the butterfly on the left and the name and session facts on the right."""

    model_name: reactive[str | None] = reactive(None, recompose=True)
    context_tokens: reactive[int | None] = reactive(None, recompose=True)

    def compose(self) -> ComposeResult:
        """Yield the butterfly and the lines of text beside it."""
        model = describe_model(self.model_name)
        context = describe_context(self.context_tokens)
        directory = shorten_home(Path.cwd(), Path.home())
        yield Butterfly()
        with Vertical(id="banner-text"):
            yield Static("agent-CLI", id="banner-name")
            yield Static("terminal agent for GGUF or API models", id="banner-tagline")
            yield Static(info_row("model", model), classes="banner-info")
            yield Static(info_row("context", context), classes="banner-info")
            yield Static(info_row("directory", directory), classes="banner-info")
