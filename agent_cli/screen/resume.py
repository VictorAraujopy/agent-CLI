"""The list that agentcli -r opens over the screen: the earlier conversations of the folder, to continue one."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from agent_cli.history.queries import SavedConversation
from agent_cli.screen.theme import MUTED_MARKUP


def option_of(conversation: SavedConversation) -> Option:
    """Return the line of a conversation in the list: when it was last touched, to the minute, and its title."""
    last_touched = conversation.updated_at[:16]
    return Option(Content.assemble((last_touched, MUTED_MARKUP), "   ", conversation.title))


class ConversationPicker(ModalScreen[SavedConversation | None]):
    """The earlier conversations of the folder, the newest first: enter continues one, esc starts a new one."""

    # Without it this screen takes the app's rule, which gives the keyboard to the message box it does not have.
    AUTO_FOCUS = "OptionList"
    BINDINGS = [("escape", "start_new", "New conversation")]

    def __init__(self, conversations: list[SavedConversation]) -> None:
        """Keep the conversations to list, in the order they are shown."""
        super().__init__()
        self.conversations = conversations

    def compose(self) -> ComposeResult:
        """Yield the title, the list and the keys that work, in one box."""
        with Vertical(id="picker"):
            yield Static("earlier conversations in this folder", id="picker-title")
            yield OptionList(*[option_of(conversation) for conversation in self.conversations])
            yield Static("↑↓ choose   enter continues   esc starts a new one", id="picker-keys")

    @on(OptionList.OptionSelected)
    def continue_chosen(self, selected: OptionList.OptionSelected) -> None:
        """Close the list with the conversation chosen."""
        self.dismiss(self.conversations[selected.option_index])

    def action_start_new(self) -> None:
        """Close the list with no conversation: the session starts a new one."""
        self.dismiss(None)
