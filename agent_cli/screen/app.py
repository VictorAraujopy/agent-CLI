"""The screen: the Textual app that reads what the user types and shows what arrives."""

from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import Mount, Unmount
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widgets import Input

from agent_cli.agent.conversation import Conversation, MessageTooLongError
from agent_cli.commands import completions, is_command, run_command
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.model import Model
from agent_cli.screen.banner import Banner
from agent_cli.screen.messages import Notice, TurnView
from agent_cli.screen.status import StatusLine
from agent_cli.screen.theme import THEME_NAME, build_theme


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
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (80, "-wide")]
    BINDINGS = [("escape", "stop_reply", "Stop the reply")]

    # What the background threads tell the screen. A thread never touches the screen: it posts one of
    # these messages, and the app handles it in its own thread. A message about a turn carries that turn,
    # so the app does not search the screen for it, which takes longer the longer the conversation is.

    @dataclass
    class ModelLoaded(Message):
        """The model of this name was opened and is ready to reply, with a context of this many tokens."""

        name: str
        model: Model
        context_tokens: int

    @dataclass
    class ModelFailedToLoad(Message):
        """The model of this name could not be opened."""

        name: str
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

    def __init__(self, conversation: Conversation, catalog: Catalog) -> None:
        """Keep the conversation and the catalog of the models the session can use."""
        super().__init__()
        self.conversation = conversation
        self.catalog = catalog
        # None until the background thread opens the first model.
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
        """Lock the message box and start opening the first model of the catalog in the background."""
        self.open_model(self.catalog.names()[0])

    def open_model(self, name: str) -> None:
        """Lock the message box and open the named model in the background."""
        self.show_activity(Activity.LOADING_MODEL)
        self.load_model(name)

    @work(thread=True)
    def load_model(self, name: str) -> None:
        """Open the model in a background thread and tell the screen whether it worked."""
        try:
            model = self.catalog.open(name)
        except Exception as error:
            self.post_message(self.ModelFailedToLoad(name, str(error)))
            return
        self.post_message(self.ModelLoaded(name, model, model.context_limit()))

    @on(ModelLoaded)
    async def use_model(self, loaded: ModelLoaded) -> None:
        """Keep the model that was opened, show its name and context and unlock the box, or close if asked."""
        replaced_a_model = self.model is not None
        self.model = loaded.model
        if self.close_when_loaded:
            self.exit()
            return
        banner = self.query_one(Banner)
        banner.model_name = loaded.name
        banner.context_tokens = loaded.context_tokens
        self.query_one(StatusLine).context_used = self.conversation.tokens_in_context / loaded.context_tokens
        self.query_one(Input).suggester = SuggestFromList(completions(loaded.model, self.catalog))
        if replaced_a_model:
            await self.show_notice(f"model: {loaded.name}   context: {loaded.context_tokens} tokens", tone="muted")
        self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(ModelFailedToLoad)
    async def report_model_not_opened(self, failed: ModelFailedToLoad) -> None:
        """Say why the model did not open and stay on the one in use; with none in use, close with the reason."""
        if self.model is None:
            self.exit(return_code=1, message=f"agent-CLI: could not load the model: {failed.reason}")
        elif self.close_when_loaded:
            self.exit()
        else:
            await self.show_notice(f"could not open {failed.name}: {failed.reason}", tone="error")
            self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(Input.Changed)
    def mark_command(self, changed: Input.Changed) -> None:
        """Color the message box as a command while what is typed in it is one."""
        changed.input.set_class(is_command(changed.value), "command")

    @on(Input.Submitted)
    async def send_message(self, submitted: Input.Submitted) -> None:
        """Show the message the user typed and start the turn that replies to it."""
        user_text = submitted.value.strip()
        # Only while the session waits for a message: a second enter can arrive before the box is locked.
        if self.model is None or self.activity is not Activity.WAITING_FOR_MESSAGE or not user_text:
            return
        submitted.input.clear()
        if is_command(user_text):
            await self.obey_command(user_text, self.model)
            return
        turn = TurnView(user_text)
        await self.query_one("#conversation-content").mount(turn)
        # Brings the end of the conversation into view and keeps it there while the reply grows, until
        # the user scrolls up.
        self.query_one("#conversation").anchor()
        self.show_activity(Activity.GENERATING)
        self.run_turn(self.model, turn, user_text)

    async def obey_command(self, text: str, model: Model) -> None:
        """Run the command, show what it says and open the model it asks for, if any."""
        outcome = run_command(text, model, self.catalog)
        await self.show_notice(outcome.text, tone="warning" if outcome.failed else "muted")
        self.query_one("#conversation").anchor()
        if outcome.model_to_open is not None:
            self.open_model(outcome.model_to_open)

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
        if self.activity in (Activity.LOADING_MODEL, Activity.CLOSING):
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
