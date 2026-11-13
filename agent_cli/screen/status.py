"""The status line: how full the model's context is, and the keys that work."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from agent_cli.screen.theme import MUTED_MARKUP, UNLIT_MARKUP

CONTEXT_LABEL = "context"
KEY_HINTS = "ctrl+q quit"

GAUGE_SEGMENT = "▊"
GAUGE_SEGMENTS = 16

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
MEDIUM_FROM = 0.60
HIGH_FROM = 0.85


def usage_level(context_used: float) -> str:
    """Return how full the context is, as one of the three levels the stylesheet colors."""
    if context_used >= HIGH_FROM:
        return HIGH
    if context_used >= MEDIUM_FROM:
        return MEDIUM
    return LOW


def gauge(context_used: float) -> str:
    """Return the bar of segments, lit up to the fraction of the context that is in use."""
    fraction = min(max(context_used, 0.0), 1.0)
    lit_count = round(fraction * GAUGE_SEGMENTS)
    lit = GAUGE_SEGMENT * lit_count
    unlit = GAUGE_SEGMENT * (GAUGE_SEGMENTS - lit_count)
    return f"{lit}[{UNLIT_MARKUP}]{unlit}[/]"


def context_text(context_used: float, tokens_in_context: int) -> str:
    """Return the label, the gauge, the percentage of the context in use and the tokens the conversation takes."""
    percent = round(context_used * 100)
    tokens = f"[{MUTED_MARKUP}]{tokens_in_context} tokens[/]"
    return f"[{MUTED_MARKUP}]{CONTEXT_LABEL}[/]  {gauge(context_used)}  {percent}%  {tokens}"


def key_hints(reply_in_progress: bool) -> str:
    """Return the keys that work now: esc stops a reply, so it only shows while one is being written."""
    if reply_in_progress:
        return f"esc stop  {KEY_HINTS}"
    return KEY_HINTS


class StatusLine(Horizontal):
    """The line under the message box: the context in use on the left, the keys on the right."""

    context_used: reactive[float] = reactive(0.0, recompose=True)
    tokens_in_context: reactive[int] = reactive(0, recompose=True)
    reply_in_progress: reactive[bool] = reactive(False, recompose=True)

    def compose(self) -> ComposeResult:
        """Yield the context gauge with its tokens, and the key hints."""
        level = usage_level(self.context_used)
        context = context_text(self.context_used, self.tokens_in_context)
        yield Static(context, id="status-context", classes=level)
        yield Static(key_hints(self.reply_in_progress), id="status-keys")
