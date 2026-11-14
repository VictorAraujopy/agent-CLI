"""Tests for the screen: the model loaded in the background, the turn each message starts, and how a turn ends."""

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from textual.dom import DOMNode
from textual.pilot import Pilot
from textual.widgets import Input, Static

from agent_cli.agent.conversation import Conversation
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.generation import LocalModel
from agent_cli.inference.loader import load_model
from agent_cli.inference.model import ContextFullError, Model
from agent_cli.screen.app import Activity, AgentCliApp
from agent_cli.screen.banner import Banner
from agent_cli.screen.messages import Notice, Reply, Thinking, TurnView, thinking_and_answer
from agent_cli.screen.status import StatusLine

SYSTEM_PROMPT = "Be brief."
MODEL_NAME = "test-model"
SECONDS_TO_WAIT = 5


class ScriptedModel(Model):
    """A model for the tests: its tokens are characters, its reply is fixed, and a test can hold each piece back."""

    def __init__(self, reply_pieces: list[str], context_limit: int = 1000, hold_pieces: bool = False) -> None:
        """Keep the pieces every reply will have, the limit, and whether each piece waits for the test to allow it."""
        self.reply_pieces = reply_pieces
        self.limit = context_limit
        self.hold_pieces = hold_pieces
        self.pieces_allowed = threading.Semaphore(0)
        self.failure_after_the_pieces: Exception | None = None
        self.cut = False
        self.reply_started = threading.Event()
        self.stop_asked = threading.Event()
        self.conversations_received: list[list[dict[str, str]]] = []
        self.tokens = 0
        self.thinking_option = "on"
        self.was_closed = False

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Refuse a conversation that passes the limit, or yield the pieces, then fail if a failure was set."""
        self.stop_asked.clear()
        self.conversations_received.append(messages)
        conversation_size = sum(len(message["content"]) for message in messages)
        if conversation_size > self.limit:
            raise ContextFullError(f"{conversation_size} characters do not fit in {self.limit}")
        self.tokens = conversation_size
        self.reply_started.set()
        for piece in self.reply_pieces:
            if self.hold_pieces:
                self.pieces_allowed.acquire(timeout=SECONDS_TO_WAIT)
            if self.stop_asked.is_set():
                return
            self.tokens += len(piece)
            yield piece
        if self.failure_after_the_pieces is not None:
            raise self.failure_after_the_pieces

    def allow_piece(self) -> None:
        """Let a reply that is being held write its next piece."""
        self.pieces_allowed.release()

    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices of a model whose thinking is only turned on or off."""
        return ("on", "off")

    def thinking(self) -> str:
        """Return the choice in use."""
        return self.thinking_option

    def set_thinking(self, option: str) -> None:
        """Keep the choice, for the test to look at."""
        self.thinking_option = option

    def close(self) -> None:
        """Remember that the model was closed."""
        self.was_closed = True

    def stop(self) -> None:
        """End the reply at its next piece, also when that piece is being held."""
        self.stop_asked.set()
        self.pieces_allowed.release()

    def reply_was_cut(self) -> bool:
        """Tell what the test set: whether the last reply counts as cut by a full context."""
        return self.cut

    def tokens_in_use(self) -> int:
        """Return the characters of the last conversation and of the reply written so far."""
        return self.tokens

    def context_limit(self) -> int:
        """Return the limit given when the model was made."""
        return self.limit


def session_with(model: Model) -> AgentCliApp:
    """Return the screen of a session whose model opens at once."""
    return AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog({MODEL_NAME: lambda: model}))


def message_box(app: AgentCliApp) -> Input:
    """Return the box the user types the message in."""
    return app.query_one(Input)


def turns_on_screen(app: AgentCliApp) -> list[TurnView]:
    """Return the turns the conversation area shows, oldest first."""
    return list(app.query(TurnView))


def notices_on_screen(app: AgentCliApp) -> list[str]:
    """Return the text of every notice the conversation area shows, oldest first."""
    return [str(notice.render()) for notice in app.query(Notice)]


def text_drawn_in(reply: Reply) -> str:
    """Return the text the reply shows on the screen, one line for each of its blocks."""
    return "\n".join(str(block.render()) for block in reply.children)


def tasks_drawing_replies() -> list[asyncio.Task[Any]]:
    """Return the background tasks with which Textual draws the pieces of a reply, the ones that have not ended."""
    return [task for task in asyncio.all_tasks() if not task.done() and "MarkdownStream" in repr(task.get_coro())]


def noting_each_call(search: Callable[..., Any], searches: list[str]) -> Callable[..., Any]:
    """Return a search of the screen for widgets changed to note its name in the list each time it is called."""

    def note_and_search(*selector: Any) -> Any:
        """Note the search, then do it."""
        searches.append(search.__name__)
        return search(*selector)

    return note_and_search


def key_hints_on_screen(app: AgentCliApp) -> str:
    """Return the keys the status line shows."""
    return str(app.query_one("#status-keys", Static).render())


async def wait_until(condition: Callable[[], bool], pilot: Pilot) -> None:
    """Let the screen run until the condition holds, failing the test when it takes too long."""
    give_up_at = time.monotonic() + SECONDS_TO_WAIT
    while not condition():
        if time.monotonic() > give_up_at:
            pytest.fail("the screen did not reach the state the test waits for")
        await pilot.pause(0.01)


async def wait_for_message_box(app: AgentCliApp, pilot: Pilot) -> None:
    """Let the screen run until the message box is unlocked: the model is loaded and no turn is running."""
    await wait_until(lambda: not message_box(app).disabled, pilot)


async def send(text: str, app: AgentCliApp, pilot: Pilot) -> None:
    """Put the text in the message box and press enter."""
    message_box(app).value = text
    await pilot.press("enter")


async def test_message_box_is_locked_while_the_model_loads() -> None:
    """Until the model is open the box cannot be typed in, says it is loading, and the banner names no model."""
    model_may_open = threading.Event()

    def open_slowly() -> Model:
        """Stand in for a slow load: hand the model over only when the test allows it."""
        model_may_open.wait(SECONDS_TO_WAIT)
        return ScriptedModel(["ok"])

    app = AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog({MODEL_NAME: open_slowly}))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert message_box(app).disabled
        assert message_box(app).placeholder == Activity.LOADING_MODEL.value
        assert app.query_one(Banner).model_name is None
        assert app.query_one(Banner).context_tokens is None
        model_may_open.set()
        await wait_for_message_box(app, pilot)


async def test_loaded_model_unlocks_the_message_box_and_is_named_in_the_banner() -> None:
    """Once the model is open the box takes the keyboard, and the banner shows its name and its context."""
    app = session_with(ScriptedModel(["ok"], context_limit=5120))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        assert app.focused is message_box(app)
        assert message_box(app).placeholder == Activity.WAITING_FOR_MESSAGE.value
        assert app.query_one(Banner).model_name == MODEL_NAME
        await pilot.pause()
        facts_on_the_banner = [str(fact.render()) for fact in app.query(".banner-info")]
        assert any(MODEL_NAME in fact for fact in facts_on_the_banner)
        assert any("5120 tokens" in fact for fact in facts_on_the_banner)


async def test_model_that_fails_to_load_closes_the_screen_with_the_reason(capsys: pytest.CaptureFixture[str]) -> None:
    """A model that cannot be opened ends the session with an error code and the reason left on the terminal."""

    def fail_to_open() -> Model:
        """Stand in for a load that fails, as it does when an RPC server is off."""
        raise ConnectionError("the server did not answer")

    app = AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog({MODEL_NAME: fail_to_open}))
    async with app.run_test() as pilot:
        await wait_until(lambda: app.return_code is not None, pilot)
    assert app.return_code == 1
    written_on_the_terminal = " ".join(capsys.readouterr().err.split())
    assert "could not load the model: the server did not answer" in written_on_the_terminal


async def test_quitting_while_the_model_loads_waits_for_the_load_to_end() -> None:
    """A quit asked during the load keeps the screen up, says so, and closes it once the model is open."""
    model_may_open = threading.Event()

    def open_slowly() -> Model:
        """Stand in for a slow load: hand the model over only when the test allows it."""
        model_may_open.wait(SECONDS_TO_WAIT)
        return ScriptedModel(["ok"])

    app = AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog({MODEL_NAME: open_slowly}))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert app.return_code is None
        assert message_box(app).placeholder == Activity.CLOSING.value
        assert message_box(app).disabled
        model_may_open.set()
        await wait_until(lambda: app.return_code is not None, pilot)
    assert app.return_code == 0


async def test_enter_sends_the_message_and_shows_the_reply() -> None:
    """A message typed and sent appears with the model's reply under it, and the conversation keeps both."""
    model = ScriptedModel(["Hel", "lo ", "**there**"])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await pilot.press("h", "i", "enter")
        await wait_for_message_box(app, pilot)
        (turn,) = turns_on_screen(app)
        assert turn.user_text == "hi"
        assert turn.reply.source == "Hello **there**"
        assert notices_on_screen(app) == []
    asked = {"role": "user", "content": "hi"}
    assert model.conversations_received == [[{"role": "system", "content": SYSTEM_PROMPT}, asked]]
    assert app.conversation.turns[0].messages == [asked, {"role": "assistant", "content": "Hello **there**"}]


async def test_sent_message_leaves_the_box_empty_and_ready_for_the_next_one() -> None:
    """After a turn the box is empty, unlocked and holds the keyboard, so the next message can be typed at once."""
    app = session_with(ScriptedModel(["ok"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("first", app, pilot)
        await wait_for_message_box(app, pilot)
        assert message_box(app).value == ""
        assert app.focused is message_box(app)
        await pilot.press("s", "e", "c", "o", "n", "d", "enter")
        await wait_for_message_box(app, pilot)
        assert [turn.user_text for turn in turns_on_screen(app)] == ["first", "second"]


async def test_message_with_only_spaces_starts_no_turn() -> None:
    """Enter on an empty box, or on spaces alone, sends nothing to the model and shows nothing."""
    model = ScriptedModel(["ok"])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("   ", app, pilot)
        await pilot.pause()
        assert turns_on_screen(app) == []
        assert not message_box(app).disabled
    assert model.conversations_received == []


async def test_square_brackets_the_user_typed_are_shown_as_text() -> None:
    """Text that looks like a style tag is shown as it was typed, in the message and in a notice."""
    model = ScriptedModel(["ok"], hold_pieces=True)
    model.failure_after_the_pieces = RuntimeError("bad [value]")
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("[bold]not a tag[/]", app, pilot)
        await wait_until(lambda: model.reply_started.is_set(), pilot)
        shown_message = str(app.query_one(".user-text", Static).render())
        model.allow_piece()
        await wait_for_message_box(app, pilot)
        assert shown_message == "[bold]not a tag[/]"
        assert notices_on_screen(app) == ["the reply failed: bad [value]"]


async def test_text_between_angle_brackets_in_a_reply_is_shown() -> None:
    """A tag or a type the model writes between < and > stays on the screen, instead of vanishing as HTML."""
    app = session_with(ScriptedModel(["Close the <div> tag. ", "A vector<int> holds ints."]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        assert text_drawn_in(turns_on_screen(app)[0].reply) == "Close the <div> tag. A vector<int> holds ints."


@pytest.mark.parametrize(
    ("written", "thinking", "answer"),
    [
        ("Just an answer.", "", "Just an answer."),
        ("<think>\nfirst the plan", "first the plan", ""),
        ("<think>\nthe plan\n</think>\n\nThe answer.", "the plan", "The answer."),
        ("<think>\n\n</think>\n\nThe answer.", "", "The answer."),
        ("<thi", "", ""),
        ("<think>\nthe plan\n</thi", "the plan", ""),
        ("Use <think> in the text.", "", "Use <think> in the text."),
        ("<div> is a tag.", "", "<div> is a tag."),
    ],
)
def test_what_the_model_wrote_is_split_into_thinking_and_answer(written: str, thinking: str, answer: str) -> None:
    """Thinking is what comes between the tags at the start; a tag still being written belongs to neither part."""
    assert thinking_and_answer(written) == (thinking, answer)


async def test_thinking_is_shown_apart_from_the_reply() -> None:
    """What the model thinks appears above its reply, also with a tag split in two pieces; the conversation keeps it."""
    model = ScriptedModel(["<th", "ink>\nfirst ", "the plan\n</th", "ink>\n\nThe ", "answer."])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        turn = turns_on_screen(app)[0]
        assert str(turn.thinking.render()) == "first the plan"
        assert turn.thinking.display
        assert turn.reply.source == "The answer."
    kept = app.conversation.turns[0].messages[1]["content"]
    assert kept == "<think>\nfirst the plan\n</think>\n\nThe answer."


async def test_thinking_grows_while_the_model_thinks() -> None:
    """Each piece of thinking is shown as it arrives, before any piece of the reply."""
    model = ScriptedModel(["<think>\nfirst ", "the plan", "\n</think>\n\nThe answer."], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        model.allow_piece()
        await wait_until(lambda: turns_on_screen(app)[0].thinking.shown == "first", pilot)
        model.allow_piece()
        await wait_until(lambda: turns_on_screen(app)[0].thinking.shown == "first the plan", pilot)
        assert turns_on_screen(app)[0].reply.source == ""
        model.allow_piece()
        await wait_for_message_box(app, pilot)
        assert turns_on_screen(app)[0].reply.source == "The answer."


async def test_reply_with_no_thinking_shows_no_room_for_it() -> None:
    """A model that does not think leaves the thinking hidden, so the reply sits right under the message."""
    app = session_with(ScriptedModel(["Just an answer."]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        assert not app.query_one(Thinking).display


async def test_message_box_is_locked_while_the_reply_is_written() -> None:
    """During a reply the box cannot be typed in and says so, and the status line offers esc only then."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        assert key_hints_on_screen(app) == "ctrl+q quit"
        await send("count", app, pilot)
        await wait_until(lambda: model.reply_started.is_set(), pilot)
        assert message_box(app).disabled
        assert message_box(app).placeholder == Activity.GENERATING.value
        assert key_hints_on_screen(app) == "esc stop  ctrl+q quit"
        model.allow_piece()
        model.allow_piece()
        await wait_for_message_box(app, pilot)
        assert key_hints_on_screen(app) == "ctrl+q quit"


async def test_each_piece_is_shown_as_it_arrives() -> None:
    """The reply on the screen grows piece by piece, before the model finishes it."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("count", app, pilot)
        model.allow_piece()
        await wait_until(lambda: turns_on_screen(app)[0].reply.source == "one ", pilot)
        assert message_box(app).disabled
        model.allow_piece()
        await wait_for_message_box(app, pilot)
        assert turns_on_screen(app)[0].reply.source == "one two"


async def test_finished_reply_leaves_no_task_drawing_it() -> None:
    """When a reply ends, the task that drew its pieces ends too, instead of one more left waiting after each turn."""
    app = session_with(ScriptedModel(["a ", "reply"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        assert turns_on_screen(app)[0].reply.source == "a reply"
        assert tasks_drawing_replies() == []


async def test_esc_stops_the_reply_and_keeps_what_was_written() -> None:
    """Esc asks the model to stop; the pieces already written stay on the screen and in the conversation."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("count", app, pilot)
        model.allow_piece()
        await wait_until(lambda: turns_on_screen(app)[0].reply.source == "one ", pilot)
        await pilot.press("escape")
        await wait_for_message_box(app, pilot)
        assert model.stop_asked.is_set()
        assert turns_on_screen(app)[0].reply.source == "one "
    assert app.conversation.turns[0].messages[1] == {"role": "assistant", "content": "one "}


async def test_closing_the_screen_stops_the_reply_being_written() -> None:
    """Quitting in the middle of a reply asks the model to stop, so its thread does not hold the terminal."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("count", app, pilot)
        await wait_until(lambda: model.reply_started.is_set(), pilot)
        await pilot.press("ctrl+q")
    assert model.stop_asked.is_set()


async def test_context_in_use_is_shown_after_the_reply() -> None:
    """When a turn ends the status line shows the tokens the conversation takes and their share of the context."""
    model = ScriptedModel(["a reply"], context_limit=200)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        status = app.query_one(StatusLine)
        assert status.tokens_in_context == 0
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        tokens = len(SYSTEM_PROMPT) + len("a question") + len("a reply")
        assert status.tokens_in_context == tokens
        assert status.context_used == tokens / 200
        assert f"{tokens} tokens" in str(app.query_one("#status-context", Static).render())


async def test_reply_cut_by_a_full_context_is_reported() -> None:
    """A reply that ended because the context filled is followed by a notice that says so."""
    model = ScriptedModel(["half a rep"])
    model.cut = True
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        assert turns_on_screen(app)[0].reply.source == "half a rep"
        assert notices_on_screen(app) == ["reply cut: the model's context is full"]


async def test_message_that_does_not_fit_is_given_back() -> None:
    """A message too long for the context leaves the conversation area, returns to the box and gets a notice."""
    too_long = "word " * 20
    model = ScriptedModel(["ok"], context_limit=50)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send(too_long, app, pilot)
        await wait_until(lambda: notices_on_screen(app) != [], pilot)
        assert notices_on_screen(app) == ["message not sent: it does not fit in the model's context"]
        assert turns_on_screen(app) == []
        assert not message_box(app).disabled
        assert message_box(app).value == too_long.strip()
        assert message_box(app).cursor_position == len(too_long.strip())
        assert message_box(app).selection.is_empty
    assert app.conversation.turns == []


async def test_failed_reply_is_reported_and_the_message_given_back() -> None:
    """A failure of the model takes the turn off the screen, as the conversation keeps none of it, and says why."""
    model = ScriptedModel(["some ", "text"])
    model.failure_after_the_pieces = RuntimeError("llama_decode returned -1")
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_until(lambda: notices_on_screen(app) != [], pilot)
        assert notices_on_screen(app) == ["the reply failed: llama_decode returned -1"]
        assert turns_on_screen(app) == []
        assert message_box(app).value == "a question"
        assert not message_box(app).disabled
    assert app.conversation.turns == []


async def test_failed_reply_stops_being_drawn_before_it_leaves_the_screen() -> None:
    """A reply that fails after some pieces ends the task that draws it: none is left waiting for more pieces."""
    model = ScriptedModel(["some ", "text"], hold_pieces=True)
    model.failure_after_the_pieces = RuntimeError("llama_decode returned -1")
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        model.allow_piece()
        await wait_until(lambda: turns_on_screen(app)[0].reply.source == "some ", pilot)
        # Kept here so that Python cannot discard the reply, and the task that draws it, once it leaves the screen.
        reply = turns_on_screen(app)[0].reply
        assert tasks_drawing_replies() != []
        model.allow_piece()
        await wait_until(lambda: notices_on_screen(app) != [], pilot)
        await wait_for_message_box(app, pilot)
        assert not reply.is_attached
        assert tasks_drawing_replies() == []


async def test_banner_stays_at_the_top_while_the_conversation_is_short() -> None:
    """A conversation shorter than its area keeps starting at the top, instead of being pushed to the bottom."""
    app = session_with(ScriptedModel(["a short reply"]))
    async with app.run_test(size=(100, 40)) as pilot:
        await wait_for_message_box(app, pilot)
        banner_row_before = app.query_one(Banner).region.y
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        assert app.query_one(Banner).region.y == banner_row_before


async def test_conversation_follows_the_end_of_a_long_reply() -> None:
    """A reply longer than the conversation area leaves the area scrolled to its last line."""
    app = session_with(ScriptedModel(["line\n\n"] * 60))
    async with app.run_test(size=(100, 40)) as pilot:
        await wait_for_message_box(app, pilot)
        await send("write many lines", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        conversation_area = app.query_one("#conversation")
        assert conversation_area.max_scroll_y > 0
        assert conversation_area.scroll_y == conversation_area.max_scroll_y


async def test_reply_leaves_no_empty_line_above_its_first_block_or_below_its_last() -> None:
    """The first block of a reply has no margin above it and the last none below, whatever kind of block each is."""
    app = session_with(ScriptedModel(["# Title\n\n", "some text\n\n", "the end"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("a question", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        title, text, end = turns_on_screen(app)[0].reply.children
        assert title.styles.margin.top == 0
        assert text.styles.margin.bottom == 1
        assert end.styles.margin.bottom == 0


async def test_click_on_the_conversation_leaves_the_keyboard_on_the_message_box() -> None:
    """After a click on the conversation area, what is typed still goes to the message box."""
    app = session_with(ScriptedModel(["ok"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await pilot.click("#conversation")
        await pilot.press("h", "i")
        assert message_box(app).value == "hi"


async def test_new_turn_does_not_restyle_the_turns_already_on_the_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn applies the styles to its own widgets only, so what a turn costs does not grow with the conversation."""
    app = session_with(ScriptedModel(["a ", "reply"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("first", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        widgets_of_the_first_turn = set(turns_on_screen(app)[0].walk_children(with_self=True))
        restyled: list[DOMNode] = []
        apply_styles = app.stylesheet.apply

        def apply_styles_and_record(node: DOMNode, **options: Any) -> None:
            """Record the widget whose styles Textual applies again, then apply them."""
            restyled.append(node)
            apply_styles(node, **options)

        monkeypatch.setattr(app.stylesheet, "apply", apply_styles_and_record)
        await send("second", app, pilot)
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        assert restyled != []
        assert widgets_of_the_first_turn.isdisjoint(restyled)


async def test_piece_of_a_reply_is_drawn_without_searching_the_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A piece goes straight to the reply of its turn: a search at each piece would slow a long conversation."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    searches: list[str] = []
    monkeypatch.setattr(app, "query", noting_each_call(app.query, searches))
    monkeypatch.setattr(app, "query_one", noting_each_call(app.query_one, searches))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("count", app, pilot)
        await wait_until(lambda: model.reply_started.is_set(), pilot)
        reply = turns_on_screen(app)[0].reply
        searches.clear()
        model.allow_piece()
        await wait_until(lambda: reply.source == "one ", pilot)
        assert searches == []
        model.allow_piece()
        await wait_for_message_box(app, pilot)


async def test_session_with_a_real_local_model_shows_its_reply(small_model_file: Path) -> None:
    """The small real model is loaded in the background, replies on the screen and stops when esc is pressed."""
    app = session_with(LocalModel(load_model(small_model_file)))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("Once upon a time", app, pilot)
        await wait_until(lambda: turns_on_screen(app)[0].reply.source != "", pilot)
        await pilot.press("escape")
        await wait_for_message_box(app, pilot)
        shown_reply = turns_on_screen(app)[0].reply.source
        assert shown_reply != ""
        assert app.query_one(StatusLine).tokens_in_context > 0
    assert app.conversation.turns[0].messages[1] == {"role": "assistant", "content": shown_reply}


async def test_enter_pressed_twice_sends_the_message_once() -> None:
    """A second enter that arrives before the box is locked starts no second turn."""
    model = ScriptedModel(["one ", "two"], hold_pieces=True)
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await pilot.press("h", "i")
        message_box(app).post_message(Input.Submitted(message_box(app), "hi"))
        message_box(app).post_message(Input.Submitted(message_box(app), "hi"))
        await wait_until(lambda: model.reply_started.is_set(), pilot)
        model.allow_piece()
        model.allow_piece()
        await wait_for_message_box(app, pilot)
        await pilot.pause()
        assert [turn.user_text for turn in turns_on_screen(app)] == ["hi"]
    assert len(model.conversations_received) == 1


def session_with_models(models: dict[str, Model]) -> AgentCliApp:
    """Return the screen of a session that can use the models given, by name, starting on the first."""
    how_to_open = {name: (lambda model=model: model) for name, model in models.items()}
    return AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog(how_to_open))


async def test_command_is_run_by_the_screen_and_never_reaches_the_model() -> None:
    """Text that starts with a slash is a command: its answer is shown as a notice and no turn starts."""
    model = ScriptedModel(["ok"])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("/think off", app, pilot)
        await wait_until(lambda: notices_on_screen(app) == ["thinking: off"], pilot)
        assert model.thinking_option == "off"
        assert turns_on_screen(app) == []
        assert model.conversations_received == []
        assert message_box(app).value == ""
        assert not message_box(app).disabled


@pytest.mark.parametrize("text", ["/ think off", "a /think off"])
async def test_slash_followed_by_a_space_or_in_the_middle_is_a_message(text: str) -> None:
    """Only a slash at the start, with the command right after it, makes a command: the rest goes to the model."""
    model = ScriptedModel(["ok"])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send(text, app, pilot)
        await wait_until(lambda: len(model.conversations_received) == 1, pilot)
        assert model.conversations_received[0][-1] == {"role": "user", "content": text}
        assert model.thinking_option == "on"
        await wait_for_message_box(app, pilot)


async def test_message_box_is_marked_while_a_command_is_typed_and_completes_it() -> None:
    """While the text is a command the box carries the command style, and it suggests the rest of the command."""
    app = session_with(ScriptedModel(["ok"]))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await pilot.press("/", "t")
        await wait_until(lambda: message_box(app).has_class("command"), pilot)
        await wait_until(lambda: message_box(app)._suggestion == "/think", pilot)
        await pilot.press("backspace", "backspace", "h", "i")
        await wait_until(lambda: not message_box(app).has_class("command"), pilot)


async def test_unknown_command_is_reported_and_not_sent() -> None:
    """A command that does not exist gets a warning with the ones that do, and the model gets nothing."""
    model = ScriptedModel(["ok"])
    app = session_with(model)
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("/usr/bin", app, pilot)
        await wait_until(lambda: len(notices_on_screen(app)) == 1, pilot)
        assert notices_on_screen(app) == ["no such command: /usr/bin   commands: /model, /think"]
        assert app.query_one(Notice).has_class("warning")
        assert model.conversations_received == []


async def test_model_command_switches_the_model_and_keeps_the_conversation() -> None:
    """After /model the next message goes to the other model with the earlier turns, and the first one is closed."""
    first, second = ScriptedModel(["one"]), ScriptedModel(["two"], context_limit=2000)
    app = session_with_models({"first": first, "second": second})
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("hello", app, pilot)
        await wait_until(lambda: len(first.conversations_received) == 1, pilot)
        await wait_for_message_box(app, pilot)
        await send("/model 2", app, pilot)
        await wait_until(lambda: app.model is second, pilot)
        await wait_for_message_box(app, pilot)
        assert first.was_closed and not second.was_closed
        assert app.query_one(Banner).model_name == "second"
        assert app.query_one(Banner).context_tokens == 2000
        assert notices_on_screen(app) == ["opening second…", "model: second   context: 2000 tokens"]
        await send("again", app, pilot)
        await wait_until(lambda: len(second.conversations_received) == 1, pilot)
        contents = [message["content"] for message in second.conversations_received[0]]
        assert contents == [SYSTEM_PROMPT, "hello", "one", "again"]
        await wait_for_message_box(app, pilot)


async def test_model_that_fails_to_open_in_a_switch_leaves_the_session_on_the_one_in_use() -> None:
    """A switch that fails is reported, the model in use stays open and the session goes on."""
    first = ScriptedModel(["one"])

    def fail_to_open() -> Model:
        """Stand in for a model that cannot be opened."""
        raise ConnectionError("the server did not answer")

    app = AgentCliApp(Conversation(SYSTEM_PROMPT), Catalog({"first": lambda: first, "second": fail_to_open}))
    async with app.run_test() as pilot:
        await wait_for_message_box(app, pilot)
        await send("/model second", app, pilot)
        await wait_until(lambda: len(notices_on_screen(app)) == 2, pilot)
        assert notices_on_screen(app)[1] == "could not open second: the server did not answer"
        await wait_for_message_box(app, pilot)
        assert app.model is first and not first.was_closed
        assert app.return_code is None
