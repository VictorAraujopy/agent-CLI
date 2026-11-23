"""The screen: the Textual app that reads what the user types and shows what arrives."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import Mount, Unmount
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widgets import Input

from agent_cli.agent.conversation import Conversation, MessageTooLongError, ToolEnded, ToolStarted
from agent_cli.commands import completions, is_command, run_command
from agent_cli.history.queries import SavedConversation, SavedTurn
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.model import Model
from agent_cli.screen.banner import Banner
from agent_cli.screen.messages import Notice, TurnView
from agent_cli.screen.permission import PermissionAnswer, PermissionPrompt
from agent_cli.screen.resume import ConversationPicker
from agent_cli.screen.status import StatusLine
from agent_cli.screen.theme import THEME_NAME, build_theme
from agent_cli.tools.tool import ToolResult
from agent_cli.tools.toolbox import ToolUse


class Activity(Enum):
    """What the session is doing now. The value of each one is what the message box says meanwhile."""

    LOADING_MODEL = "Loading the model…"
    WAITING_FOR_MESSAGE = "Type a message"
    GENERATING = "Generating…"
    WAITING_FOR_PERMISSION = "Answer above: may the tool run?"
    CLOSING = "Closing as soon as the model is loaded…"
    CLOSING_AFTER_THE_TURN = "Closing as soon as the turn is saved…   ctrl+q closes now"


def only_while_running(handler: Callable[[Any, Any], Awaitable[None]]) -> Callable[[Any, Any], Awaitable[None]]:
    """Make a handler of what a background thread tells the screen do nothing once the app is closing."""

    # Textual stops running, takes the screen away and only then handles the messages still waiting, so a
    # message a thread posted just before the close would look for widgets that are gone.
    @wraps(handler)
    async def handle_while_running(app: "AgentCliApp", message: Any) -> None:
        if app.is_running:
            await handler(app, message)

    return handle_while_running


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
        context_limit: int

    @dataclass
    class ToolAsked(Message):
        """A reply of a turn asked for a tool, which is shown before it is allowed and run."""

        turn: TurnView
        use: ToolUse

    @dataclass
    class PermissionNeeded(Message):
        """A tool of a turn may only run if the user allows it, and the thread of the turn waits for the answer."""

        turn: TurnView
        answer: PermissionAnswer

    @dataclass
    class ToolDone(Message):
        """The tool shown last in a turn ended, run or not."""

        turn: TurnView
        result: ToolResult

    @dataclass
    class MessageTooLong(Message):
        """The user's message of a turn does not fit in the model's context, so it was not sent."""

        turn: TurnView

    @dataclass
    class ReplyFailed(Message):
        """The model failed while it read the conversation or wrote the reply of a turn."""

        turn: TurnView
        reason: str

    def __init__(self, conversation: Conversation, catalog: Catalog, resume: bool = False) -> None:
        """Keep the conversation, the catalog of the models, and whether to offer the earlier conversations first."""
        super().__init__()
        self.conversation = conversation
        self.catalog = catalog
        self.resume = resume
        self.activity = Activity.LOADING_MODEL
        self.close_when_loaded = False
        self.close_when_turn_ends = False
        # The line that asks whether a tool may run, while the user has not answered it.
        self.permission_prompt: PermissionPrompt | None = None

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
        self.open_in_background(self.catalog.names()[0])

    @on(Mount)
    async def offer_earlier_conversations(self) -> None:
        """With -r, list the earlier conversations of the folder over the screen, or say there is none."""
        if not self.resume:
            return
        earlier_conversations = self.conversation.history.conversations_in_folder()
        if not earlier_conversations:
            await self.show_notice("no earlier conversation in this folder: a new one starts", tone="muted")
            return
        self.push_screen(ConversationPicker(earlier_conversations), self.continue_conversation)

    async def continue_conversation(self, chosen: SavedConversation | None) -> None:
        """Bring back the conversation chosen, to the model and to the screen; with none, the new one goes on."""
        if chosen is None:
            return
        saved_turns = self.conversation.history.resume(chosen.id)
        self.conversation.continue_from([turn.messages for turn in saved_turns], chosen.tokens_in_context)
        for saved_turn in saved_turns:
            await self.show_saved_turn(saved_turn)
        # A model still loading shows the context when it is loaded; one already loaded shows it now.
        model_in_use = self.catalog.model_in_use
        if model_in_use is not None:
            self.show_context_in_use(model_in_use.context_limit())
        await self.show_notice(f"continuing: {chosen.title}", tone="muted")
        self.query_one("#conversation").anchor()

    async def show_saved_turn(self, saved_turn: SavedTurn) -> None:
        """Draw a turn read back from the history as it was shown: the message, the replies and their tools."""
        user_message, *later_messages = saved_turn.messages
        turn = TurnView(user_message["content"])
        await self.query_one("#conversation-content").mount(turn)
        for message in later_messages:
            if message["role"] != "assistant":
                continue
            if message["content"]:
                await turn.add_piece(message["content"])
            for call in message.get("tool_calls", []):
                use = self.conversation.toolbox.read(call["function"]["name"], call["function"]["arguments"])
                await turn.add_tool(use)
                # The file is outside the program: a call kept with no result is shown without one.
                result = saved_turn.tool_results.get(call["id"])
                if result is not None:
                    await turn.show_tool_result(result)
        await turn.finish()

    def open_in_background(self, name: str) -> None:
        """Lock the message box and start opening the named model in a background thread."""
        self.show_activity(Activity.LOADING_MODEL)
        self.open_and_report(name)

    @work(thread=True)
    def open_and_report(self, name: str) -> None:
        """Open the model, which becomes the one in use, and tell the screen whether it worked."""
        try:
            model = self.catalog.open(name)
        except Exception as error:
            self.post_message(self.ModelFailedToLoad(name, str(error)))
            return
        self.post_message(self.ModelLoaded(name, model.context_limit()))

    @on(ModelLoaded)
    @only_while_running
    async def use_model(self, loaded: ModelLoaded) -> None:
        """Show the name and the context of the model that was opened and unlock the box, or close if asked."""
        if self.close_when_loaded:
            self.exit()
            return
        banner = self.query_one(Banner)
        replaced_a_model = banner.model_name is not None
        banner.model_name = loaded.name
        banner.context_tokens = loaded.context_tokens
        self.show_context_in_use(loaded.context_tokens)
        self.query_one(Input).suggester = SuggestFromList(completions(self.catalog))
        if replaced_a_model:
            await self.show_notice(f"model: {loaded.name}   context: {loaded.context_tokens} tokens", tone="muted")
        self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(ModelFailedToLoad)
    @only_while_running
    async def report_model_not_opened(self, failed: ModelFailedToLoad) -> None:
        """Say why the model did not open and stay on the one in use; with none in use, close with the reason."""
        if self.catalog.model_in_use is None:
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
        if self.activity is not Activity.WAITING_FOR_MESSAGE or not user_text:
            return
        submitted.input.clear()
        if is_command(user_text):
            await self.obey_command(user_text)
            return
        turn = TurnView(user_text)
        await self.query_one("#conversation-content").mount(turn)
        # Brings the end of the conversation into view and keeps it there while the reply grows, until
        # the user scrolls up.
        self.query_one("#conversation").anchor()
        self.show_activity(Activity.GENERATING)
        self.run_turn(self.catalog.in_use(), turn, user_text)

    async def obey_command(self, text: str) -> None:
        """Run the command, show what it says and open the model it asks for, if any."""
        outcome = run_command(text, self.catalog)
        await self.show_notice(outcome.text, tone="warning" if outcome.failed else "muted")
        self.query_one("#conversation").anchor()
        if outcome.model_to_open is not None:
            self.open_in_background(outcome.model_to_open)

    @work(thread=True)
    def run_turn(self, model: Model, turn: TurnView, user_text: str) -> None:
        """Run the turn in a background thread, telling the screen each piece, each tool and how the turn ended."""
        ask_permission = partial(self.ask_permission, turn)
        try:
            for event in self.conversation.run_turn(model, user_text, ask_permission):
                self.post_message(self.message_about(turn, event))
        except MessageTooLongError:
            self.post_message(self.MessageTooLong(turn))
        except Exception as error:
            self.post_message(self.ReplyFailed(turn, str(error)))
        else:
            self.post_message(self.ReplyEnded(turn, self.conversation.turn_was_cut, model.context_limit()))

    def message_about(self, turn: TurnView, event: str | ToolStarted | ToolEnded) -> Message:
        """Return the message that tells the screen what happened in the turn: a piece, a tool asked for or ended."""
        if isinstance(event, ToolStarted):
            return self.ToolAsked(turn, event.use)
        if isinstance(event, ToolEnded):
            return self.ToolDone(turn, event.result)
        return self.ReplyPiece(turn, event)

    def ask_permission(self, turn: TurnView, use: ToolUse) -> bool:
        """Ask the user, from the thread of the turn, whether the tool may run, and wait for the answer."""
        answer = PermissionAnswer()
        self.post_message(self.PermissionNeeded(turn, answer))
        return answer.wait(self.conversation.stop_requested)

    async def action_quit(self) -> None:
        """Close the screen; while the model loads or a turn runs, close as soon as that ends."""
        # The thread that loads cannot be stopped, and the process only ends after it.
        if self.activity in (Activity.LOADING_MODEL, Activity.CLOSING):
            self.close_when_loaded = True
            self.show_activity(Activity.CLOSING)
            return
        # The model is closed right after the screen, so a turn still running would fail before it is saved.
        if self.activity in (Activity.GENERATING, Activity.WAITING_FOR_PERMISSION):
            self.close_when_turn_ends = True
            self.action_stop_reply()
            self.show_activity(Activity.CLOSING_AFTER_THE_TURN)
            return
        self.exit()

    def action_stop_reply(self) -> None:
        """End the turn: the reply being written, the tool running and the question waiting; what was done stays."""
        self.conversation.stop()
        model_in_use = self.catalog.model_in_use
        if model_in_use is not None:
            model_in_use.stop()
        if self.permission_prompt is not None:
            self.permission_prompt.remove()
            self.permission_prompt = None

    @on(Unmount)
    def stop_reply_when_closing(self) -> None:
        """Stop the reply when the app closes, so one still being written does not hold the terminal."""
        self.action_stop_reply()

    @on(ReplyPiece)
    @only_while_running
    async def show_piece(self, piece: ReplyPiece) -> None:
        """Add a piece of text to the end of the reply being written."""
        await piece.turn.add_piece(piece.text)

    @on(ToolAsked)
    @only_while_running
    async def show_tool(self, asked: ToolAsked) -> None:
        """Show the tool a reply asked for, under that reply."""
        await asked.turn.add_tool(asked.use)

    @on(PermissionNeeded)
    @only_while_running
    async def ask_about_tool(self, needed: PermissionNeeded) -> None:
        """Ask under the tool whether it may run, and give the keyboard to the question."""
        # A turn stopped before this message arrived has already taken the question as refused.
        if self.conversation.stop_requested.is_set():
            return
        self.permission_prompt = PermissionPrompt(needed.answer)
        await needed.turn.ask_permission(self.permission_prompt)
        self.show_activity(Activity.WAITING_FOR_PERMISSION)
        self.permission_prompt.focus()

    @on(PermissionPrompt.Answered)
    async def answer_about_tool(self, answered: PermissionPrompt.Answered) -> None:
        """Give the thread of the turn the user's answer, and take the question off the screen."""
        if answered.prompt is not self.permission_prompt:
            return
        answered.prompt.answer.give(answered.allowed)
        self.permission_prompt = None
        await answered.prompt.remove()
        self.show_activity(Activity.GENERATING)

    @on(ToolDone)
    @only_while_running
    async def show_tool_result(self, done: ToolDone) -> None:
        """Show how the tool shown last ended."""
        await done.turn.show_tool_result(done.result)

    @on(ReplyEnded)
    @only_while_running
    async def finish_turn(self, ended: ReplyEnded) -> None:
        """Draw the rest of the reply, warn if it was cut, show the context in use and unlock the message box."""
        await ended.turn.finish()
        if ended.was_cut:
            await self.show_notice("reply cut: the model's context is full", tone="muted")
        self.show_context_in_use(ended.context_limit)
        self.show_activity(Activity.WAITING_FOR_MESSAGE)

    @on(ReplyEnded)
    @on(MessageTooLong)
    @on(ReplyFailed)
    def close_if_asked_during_the_turn(self, ended: Message) -> None:
        """Close the screen when ctrl+q was pressed while the turn that just ended ran."""
        if self.close_when_turn_ends:
            self.exit()

    @on(MessageTooLong)
    @only_while_running
    async def refuse_long_message(self, refused: MessageTooLong) -> None:
        """Give back the message that does not fit in the model's context, which was not sent."""
        reason = "message not sent: it does not fit in the model's context"
        await self.give_message_back(refused.turn, reason, tone="warning")

    @on(ReplyFailed)
    @only_while_running
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

    def show_context_in_use(self, context_limit: int) -> None:
        """Show on the status line how much of the model's context the conversation takes."""
        status = self.query_one(StatusLine)
        status.tokens_in_context = self.conversation.tokens_in_context
        status.context_used = self.conversation.tokens_in_context / context_limit

    def show_activity(self, activity: Activity) -> None:
        """Keep what the session is doing and show it: in the message box, locked unless a message is awaited."""
        self.activity = activity
        message_box = self.query_one(Input)
        message_box.placeholder = activity.value
        message_box.disabled = activity is not Activity.WAITING_FOR_MESSAGE
        if not message_box.disabled:
            message_box.focus()
        turn_in_progress = activity in (Activity.GENERATING, Activity.WAITING_FOR_PERMISSION)
        self.query_one(StatusLine).reply_in_progress = turn_in_progress
