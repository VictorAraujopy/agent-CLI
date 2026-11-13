"""The screen: the Textual app that reads what the user types and shows what arrives."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import Mount, Unmount
from textual.message import Message
from textual.widgets import Input

from agent_cli.agent.conversation import Conversation, MessageTooLongError
from agent_cli.inference.model import Model
from agent_cli.screen.banner import Banner
from agent_cli.screen.messages import Notice, TurnView
from agent_cli.screen.status import StatusLine
from agent_cli.screen.theme import THEME_NAME, build_theme

WIDE_FROM_COLUMNS = 80


class Activity(Enum):
    """What the session is doing now. The value of each one is what the message box says meanwhile."""

    LOADING_MODEL = "Loading the model…"
    WAITING_FOR_MESSAGE = "Type a message"
    GENERATING = "Generating…"
    CLOSING = "Closing as soon as the model is loaded…"


class AgentCliApp(App):
    """The application that owns the terminal window."""

    CSS_PATH = "app.tcss"
    AUTO_FOCUS = "Input"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (WIDE_FROM_COLUMNS, "-wide")]
    BINDINGS = [("escape", "stop_reply", "Stop the reply")]

    # What the background threads tell the screen. A thread never touches the screen: it posts one of
    # these messages, and the app handles it in its own thread. A message about a turn carries that turn,
    # so the app does not search the screen for it, which takes longer the longer the conversation is.

    @dataclass
    class ModelLoaded(Message):
        """The model was opened and is ready to reply, with a context of this many tokens."""

        model: Model
        context_tokens: int

    @dataclass
    class ModelFailedToLoad(Message):
        """The model could not be opened."""

        reason: str

    @dataclass
    class ReplyPiece(Message):
        """The model wrote one more piece of the reply of a turn."""

        turn: TurnView
        text: str

    @dataclass
    class ReplyEnded(Message):
        """The reply ended and the conversation kept the turn: the model finished it, or it was cut or stopped."""

        turn: TurnView
        was_cut: bool
        tokens_in_context: int
        context_limit: int

    @dataclass
    class MessageTooLong(Message):
        """The user's message of a turn does not fit in the model's context, so it was not sent."""

        turn: TurnView

    @dataclass
    class ReplyFailed(Message):
        """The model failed while it read the conversation or wrote the reply of a turn."""

        turn: TurnView
        reason: str

    def __init__(self, conversation: Conversation, model_name: str, open_model: Callable[[], Model]) -> None:
        """Keep the conversation, the name to show for the model and the function that opens the model."""
        super().__init__()
        self.conversation = conversation
        self.model_name = model_name
        self.open_model = open_model
        # None until the background thread opens the model.
        self.model: Model | None = None
        self.activity = Activity.LOADING_MODEL
        self.close_when_loaded = False

    def compose(self) -> ComposeResult:
        """Yield the conversation area with the banner, the message box and the status line."""
        # can_focus=False: a click on the conversation leaves the keyboard on the message box. And a locked
        # box does not hand the focus to the conversation, which would make Textual apply the styles again
        # to every widget in it, at the start and at the end of each turn.
        with VerticalScroll(id="conversation", can_focus=False):
            with Vertical(id="conversation-content"):
                yield Banner()
        # select_on_focus=False: a message given back to the box must not be replaced by the next key typed.
        yield Input(select_on_focus=False)
        yield StatusLine()

    @on(Mount)
    def apply_theme(self) -> None:
        """Register the palette and make it the active theme."""
        self.register_theme(build_theme())
        self.theme = THEME_NAME

    @on(Mount)
    def start_loading_model(self) -> None:
        """Lock the message box and start opening the model in the background."""
        self.show_activity(Activity.LOADING_MODEL)
        self.load_model()

    @work(thread=True)
    def load_model(self) -> None:
        """Open the model in a background thread and tell the screen whether it worked."""
        try:
            model = self.open_model()
        except Exception as error:
            self.post_message(self.ModelFailedToLoad(str(error)))
            return
        self.post_message(self.ModelLoaded(model, model.context_limit()))

    @on(ModelLoaded)
    def start_session(self, loaded: ModelLoaded) -> None:
        """Keep the model that was opened, show its name and context and unlock the box, or close if asked."""
        self.model = loaded.model
        if self.close_when_loaded:
            self.exit()
            return
        banner = self.query_one(Banner)
        banner.model_name = self.model_name
        banner.context_tokens = loaded.context_tokens
        self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(ModelFailedToLoad)
    def close_without_model(self, failed: ModelFailedToLoad) -> None:
        """Close the screen and leave the reason on the terminal: there is no session without a model."""
        self.exit(return_code=1, message=f"agent-CLI: could not load the model: {failed.reason}")

    @on(Input.Submitted)
    async def send_message(self, submitted: Input.Submitted) -> None:
        """Show the message the user typed and start the turn that replies to it."""
        user_text = submitted.value.strip()
        # Only while the session waits for a message: a second enter can arrive before the box is locked.
        if self.model is None or self.activity is not Activity.WAITING_FOR_MESSAGE or not user_text:
            return
        submitted.input.clear()
        turn = TurnView(user_text)
        await self.query_one("#conversation-content").mount(turn)
        # Brings the end of the conversation into view and keeps it there while the reply grows, until
        # the user scrolls up.
        self.query_one("#conversation").anchor()
        self.show_activity(Activity.GENERATING)
        self.run_turn(self.model, turn, user_text)

    @work(thread=True)
    def run_turn(self, model: Model, turn: TurnView, user_text: str) -> None:
        """Run the turn in a background thread, telling the screen each piece of the reply and how it ended."""
        try:
            for piece in self.conversation.run_turn(model, user_text):
                self.post_message(self.ReplyPiece(turn, piece))
        except MessageTooLongError:
            self.post_message(self.MessageTooLong(turn))
        except Exception as error:
            self.post_message(self.ReplyFailed(turn, str(error)))
        else:
            tokens_in_context = self.conversation.tokens_in_context
            was_cut = model.reply_was_cut()
            self.post_message(self.ReplyEnded(turn, was_cut, tokens_in_context, model.context_limit()))

    async def action_quit(self) -> None:
        """Close the screen, or leave that for when the model is loaded, if it is still loading."""
        # The thread that loads cannot be stopped, and the process only ends after it. Without the screen,
        # what the library writes during the load would land on the terminal.
        if self.model is None:
            self.close_when_loaded = True
            self.show_activity(Activity.CLOSING)
            return
        self.exit()

    def action_stop_reply(self) -> None:
        """Ask the model to end the reply it is writing; the turn then ends as usual, with what was written."""
        if self.model is not None:
            self.model.stop()

    @on(Unmount)
    def stop_reply_when_closing(self) -> None:
        """Stop the reply when the app closes, so one still being written does not hold the terminal."""
        self.action_stop_reply()

    @on(ReplyPiece)
    async def show_piece(self, piece: ReplyPiece) -> None:
        """Add a piece of text to the end of the reply being written."""
        await piece.turn.add_piece(piece.text)

    @on(ReplyEnded)
    async def finish_turn(self, ended: ReplyEnded) -> None:
        """Draw the rest of the reply, warn if it was cut, show the context in use and unlock the message box."""
        await ended.turn.finish()
        if ended.was_cut:
            await self.show_notice("reply cut: the model's context is full", tone="muted")
        status = self.query_one(StatusLine)
        status.tokens_in_context = ended.tokens_in_context
        status.context_used = ended.tokens_in_context / ended.context_limit
        self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(MessageTooLong)
    async def refuse_long_message(self, refused: MessageTooLong) -> None:
        """Give back the message that does not fit in the model's context, which was not sent."""
        reason = "message not sent: it does not fit in the model's context"
        await self.give_message_back(refused.turn, reason, tone="warning")

    @on(ReplyFailed)
    async def report_failed_reply(self, failed: ReplyFailed) -> None:
        """Give back the message whose reply failed, with the reason of the failure."""
        await self.give_message_back(failed.turn, f"the reply failed: {failed.reason}", tone="error")

    async def give_message_back(self, turn: TurnView, reason: str, tone: str) -> None:
        """Take the turn the conversation did not keep off the screen, put its text in the message box and say why."""
        # A reply that failed after some pieces is still being drawn: that has to end before the turn leaves.
        await turn.finish()
        await turn.remove()
        await self.show_notice(reason, tone)
        self.show_activity(Activity.WAITING_FOR_MESSAGE)
        message_box = self.query_one(Input)
        message_box.value = turn.user_text
        message_box.action_end()

    async def show_notice(self, text: str, tone: str) -> None:
        """Add a line of the screen's own to the end of the conversation."""
        await self.query_one("#conversation-content").mount(Notice(text, tone))

    def show_activity(self, activity: Activity) -> None:
        """Keep what the session is doing and show it: in the message box, locked unless a message is awaited."""
        self.activity = activity
        message_box = self.query_one(Input)
        message_box.placeholder = activity.value
        message_box.disabled = activity is not Activity.WAITING_FOR_MESSAGE
        if not message_box.disabled:
            message_box.focus()
        self.query_one(StatusLine).reply_in_progress = activity is Activity.GENERATING
