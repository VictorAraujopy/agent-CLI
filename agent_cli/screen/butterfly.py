"""The butterfly of the banner, which flaps its wings from time to time."""

from rich.style import Style
from rich.text import Text
from textual import on
from textual.events import Mount
from textual.timer import Timer
from textual.widgets import Static

from agent_cli.screen.art import BUTTERFLY_FOLDED, BUTTERFLY_HALF, BUTTERFLY_OPEN
from agent_cli.screen.theme import CRIMSON, PAPER_WHITE

SECONDS_BETWEEN_FLAPS = 4.0
SECONDS_PER_FLAP_FRAME = 0.1
FLAP_FRAMES = (BUTTERFLY_HALF, BUTTERFLY_FOLDED, BUTTERFLY_HALF, BUTTERFLY_OPEN)

WING_COLOR = CRIMSON
BODY_COLOR = PAPER_WHITE
BODY_CHARACTERS = "|\\/"


def butterfly_text(rows: tuple[str, ...]) -> Text:
    """Return the drawing as text, with the body and antennae in one color and the wings in another."""
    text = Text()
    drawing = "\n".join(rows)
    for character in drawing:
        color = BODY_COLOR if character in BODY_CHARACTERS else WING_COLOR
        text.append(character, Style(color=color))
    return text


class Butterfly(Static):
    """The butterfly drawing, with wings open except during a short flap."""

    def __init__(self) -> None:
        """Start with the wings open and no flap in progress."""
        super().__init__(butterfly_text(BUTTERFLY_OPEN))
        self.next_flap_frame = 0
        self.flap_timer: Timer | None = None

    @on(Mount)
    def schedule_flaps(self) -> None:
        """Start the timer that makes the butterfly flap from time to time."""
        self.set_interval(SECONDS_BETWEEN_FLAPS, self.start_flap)

    def start_flap(self) -> None:
        """Start one flap: a short timer that shows its frames one after the other."""
        if self.flap_timer is not None:
            self.flap_timer.stop()
        self.next_flap_frame = 0
        self.flap_timer = self.set_interval(SECONDS_PER_FLAP_FRAME, self.show_next_flap_frame)

    def show_next_flap_frame(self) -> None:
        """Show the next frame of the flap in progress, and stop its timer after the last one."""
        frame = FLAP_FRAMES[self.next_flap_frame]
        self.update(butterfly_text(frame))
        self.next_flap_frame += 1
        flap_is_over = self.next_flap_frame == len(FLAP_FRAMES)
        if flap_is_over and self.flap_timer is not None:
            self.flap_timer.stop()
