"""The color palette of the screen, as a Textual theme."""

from textual.theme import Theme

THEME_NAME = "crimson"

PAPER_WHITE = "#e8e3d9"
CRIMSON = "#a4112a"

MUTED_MARKUP = "$foreground 55%"
UNLIT_MARKUP = "$foreground 15%"


def build_theme() -> Theme:
    """Build the Textual theme that gives each palette color its role on the screen."""
    return Theme(
        name=THEME_NAME,
        dark=True,
        background="#0b0b0d",
        surface="#141417",
        panel="#1c1c21",
        foreground=PAPER_WHITE,
        primary=CRIMSON,
        secondary="#5e0b19",
        accent="#d7263d",
        success="#5fbf6b",
        warning="#d9a441",
        error="#ff4d5e",
    )
