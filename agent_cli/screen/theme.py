"""The color palette of the screen, as a Textual theme."""

from textual.theme import Theme

THEME_NAME = "crimson"

NEAR_BLACK = "#0b0b0d"
CHARCOAL = "#141417"
GRAPHITE = "#1c1c21"
PAPER_WHITE = "#e8e3d9"
CRIMSON = "#a4112a"
DARK_CRIMSON = "#5e0b19"
BRIGHT_RED = "#d7263d"
GREEN = "#5fbf6b"
AMBER = "#d9a441"
SIGNAL_RED = "#ff4d5e"

MUTED_MARKUP = "$foreground 55%"
UNLIT_MARKUP = "$foreground 15%"


def build_theme() -> Theme:
    """Build the Textual theme that gives each palette color its role on the screen."""
    return Theme(
        name=THEME_NAME,
        dark=True,
        background=NEAR_BLACK,
        surface=CHARCOAL,
        panel=GRAPHITE,
        foreground=PAPER_WHITE,
        primary=CRIMSON,
        secondary=DARK_CRIMSON,
        accent=BRIGHT_RED,
        success=GREEN,
        warning=AMBER,
        error=SIGNAL_RED,
    )
