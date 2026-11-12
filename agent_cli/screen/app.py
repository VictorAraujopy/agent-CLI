"""The screen: the Textual app that reads what the user types and shows what arrives."""
from textual import on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import Mount
from textual.widgets import Input

from agent_cli.screen.banner import Banner
from agent_cli.screen.status import StatusLine
from agent_cli.screen.theme import THEME_NAME, build_theme

WIDE_FROM_COLUMNS = 80

class AgentCliApp(App):
    """the aplication that onws the terminal window."""

    CSS_PATH = "app.tcss"
    AUTO_FOCUS = "Input"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (WIDE_FROM_COLUMNS, "-wide")]

    @on(Mount)
    def apply_theme(self) -> None:
        """Register the palette and make it the active theme."""
        self.register_theme(build_theme())
        self.theme = THEME_NAME

    def compose(self) -> ComposeResult:
        """yield the widgets"""


        with VerticalScroll(id="conversation"):
            yield Banner()

        yield Input(placeholder="Type a message")
        yield StatusLine()


def main() -> None:

    app = AgentCliApp()

    app.run()

if __name__ == "__main__":
    main()
