"""Tests for the conversation: the messages it keeps, the cut of the oldest turns, and each turn it runs."""

import json
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from agent_cli.agent.conversation import (
    Conversation,
    MessageTooLongError,
    RecallFailed,
    Recalled,
    ToolEnded,
    ToolStarted,
    characters_in,
    reply_message,
)
from agent_cli.history.queries import History
from agent_cli.inference.generation import LocalModel, open_local_model
from agent_cli.inference.model import ContextFullError, Message, Model, ToolCall
from agent_cli.rag.recall import with_recalled
from agent_cli.tools.tool import Ending
from agent_cli.tools.toolbox import Toolbox, ToolUse
from scripted_model import ScriptedModel

SYSTEM_PROMPT = "Be brief."
SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}


@pytest.fixture(scope="module")
def local_model(small_model_file: Path) -> Iterator[LocalModel]:
    """Serve the small real model once for the tests that run the conversation on it, and end its server after them."""
    model = open_local_model(small_model_file)
    yield model
    model.close()


class FakeModel(Model):
    """A model for the tests: its tokens are characters, it refuses what passes its limit, and its reply is fixed."""

    def __init__(self, context_limit: int, reply_pieces: list[str], failure: Exception | None = None) -> None:
        """Keep the limit, the pieces every reply will have, and an error to raise instead of them, if any."""
        self.limit = context_limit
        self.reply_pieces = reply_pieces
        self.failure = failure
        self.conversations_received: list[list[Message]] = []
        self.tokens = 0
        self.events: list[str] = []

    def reply(self, messages: list[Message], tools: Sequence[dict[str, Any]] = ()) -> Iterator[str]:
        """Record the conversation, refuse it when its characters pass the limit, or yield the pieces."""
        self.conversations_received.append(messages)
        conversation_size = sum(len(message["content"]) for message in messages)
        if conversation_size > self.limit:
            raise ContextFullError(f"{conversation_size} characters do not fit in {self.limit}")
        if self.failure is not None:
            raise self.failure
        self.tokens = conversation_size
        try:
            for piece in self.reply_pieces:
                self.tokens += len(piece)
                yield piece
        except GeneratorExit:
            self.events.append("reply stopped")
            raise

    def tool_calls(self) -> list[ToolCall]:
        """Return no calls: the fake only writes text."""
        return []

    def tokens_in_use(self) -> int:
        """Return the characters of the last conversation and of the reply written so far."""
        self.events.append("tokens counted")
        return self.tokens

    def thinking_options(self) -> tuple[str, ...]:
        """Return no choices: the fake does not think."""
        return ()

    def thinking(self) -> str:
        """Return no choice: the fake does not think."""
        return ""

    def set_thinking(self, option: str) -> None:
        """Do nothing: the fake does not think."""

    def close(self) -> None:
        """Do nothing: the fake holds nothing."""

    def stop(self) -> None:
        """Do nothing: the tests end the fake's reply by closing it."""

    def reply_was_cut(self) -> bool:
        """Tell that the last reply was not cut: the fake always finishes its reply."""
        return False

    def context_limit(self) -> int:
        """Return the limit given when the model was made."""
        return self.limit


def history_in_a_new_folder() -> History:
    """Return a history kept in a new folder of the temporary folder of the tests."""
    folder = Path(tempfile.mkdtemp())
    return History(folder / "history.db", folder)


def new_conversation(system_prompt: str = SYSTEM_PROMPT) -> Conversation:
    """Return a conversation for turns that run no tool, so its toolbox points at no real folder."""
    return Conversation(system_prompt, Toolbox(Path("no-tool-runs-here")), history_in_a_new_folder(), recall=None)


def conversation_in(folder: Path) -> Conversation:
    """Return a conversation whose tools act from the folder, with its history kept in another one."""
    history_folder = Path(tempfile.mkdtemp())
    return Conversation(SYSTEM_PROMPT, Toolbox(folder), History(history_folder / "history.db", folder), recall=None)


def no_permission_asked(use: ToolUse) -> bool:
    """Stand in for the user in turns that run no tool: being asked fails the test."""
    raise AssertionError(f"no tool should ask for permission, but {use.name} did")


def exchange(question: str, answer: str) -> list[dict[str, str]]:
    """Return the messages of one turn: the user's question and the assistant's answer."""
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def user(text: str) -> dict[str, str]:
    """Return a message of the user."""
    return {"role": "user", "content": text}


def exchange_of_size(characters: int) -> list[dict[str, str]]:
    """Return the messages of one turn holding exactly this many characters of text."""
    return exchange("q", "a" * (characters - 1))


def conversation_with_turns(sizes: list[int], tokens_in_context: int) -> Conversation:
    """Return a conversation with one turn of each size, in characters, and the tokens the model counted last."""
    conversation = new_conversation()
    for size in sizes:
        conversation.add_turn(exchange_of_size(size), tokens_in_context)
    return conversation


def test_new_conversation_holds_only_the_system_prompt() -> None:
    """Before any turn, the only message is the system prompt, and no token is counted."""
    conversation = new_conversation()
    assert conversation.messages() == [SYSTEM_MESSAGE]
    assert conversation.tokens_in_context == 0


def test_tokens_in_context_are_the_last_count_of_the_model() -> None:
    """After each turn, the tokens in context are what the model counted then, even when the count went down."""
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=900)
    conversation.add_turn(exchange("q2", "a2"), tokens_in_context_after_reply=500)
    assert conversation.tokens_in_context == 500


def test_conversation_is_full_only_above_80_percent_of_the_context() -> None:
    """At exactly 80% of the context the conversation is not full yet; one token more and it is."""
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=800)
    assert not conversation.is_full(context_limit=1000)
    conversation.add_turn(exchange("q2", "a2"), tokens_in_context_after_reply=801)
    assert conversation.is_full(context_limit=1000)


def test_cut_drops_the_oldest_turns_until_half_of_the_context() -> None:
    """The cut takes whole turns from the start, until the estimate of the rest is at most 50% of the context."""
    # 1000 characters (9 of them the system prompt) hold 900 tokens. Without the first two turns,
    # 409 characters are left, about 368 tokens.
    conversation = conversation_with_turns([291, 300, 200, 200], tokens_in_context=900)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange_of_size(200), *exchange_of_size(200)]
    assert conversation.tokens_in_context == 368


def test_cut_stops_as_soon_as_the_rest_takes_half_of_the_context() -> None:
    """A rest of exactly 50% of the context is kept: the cut does not drop one turn more."""
    # 1000 characters hold 1000 tokens, and without the first turn 500 are left.
    conversation = conversation_with_turns([500, 491], tokens_in_context=1000)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange_of_size(491)]


def test_system_prompt_keeps_its_share_of_the_tokens_after_a_cut() -> None:
    """A long system prompt still counts once the oldest turns leave, so the cut goes on until the estimate fits."""
    conversation = new_conversation("s" * 400)
    conversation.add_turn(exchange_of_size(300), tokens_in_context_after_reply=700)
    conversation.add_turn(exchange_of_size(300), tokens_in_context_after_reply=1000)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.turns == []
    assert conversation.tokens_in_context == 400


def test_cut_depends_only_on_the_last_count_of_the_model() -> None:
    """Earlier counts change nothing, not even one that went down, as when a model drops its old reasoning."""
    count_went_down = new_conversation()
    count_went_down.add_turn(exchange_of_size(491), tokens_in_context_after_reply=950)
    count_went_down.add_turn(exchange_of_size(500), tokens_in_context_after_reply=900)
    count_went_up = new_conversation()
    count_went_up.add_turn(exchange_of_size(491), tokens_in_context_after_reply=100)
    count_went_up.add_turn(exchange_of_size(500), tokens_in_context_after_reply=900)
    count_went_down.cut_oldest_turns(context_limit=1000)
    count_went_up.cut_oldest_turns(context_limit=1000)
    assert count_went_down.messages() == count_went_up.messages()


def test_cut_never_drops_the_system_prompt() -> None:
    """Even when every turn has to go, the system prompt stays."""
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=5000)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.messages() == [SYSTEM_MESSAGE]


def test_turn_sends_the_whole_conversation_and_keeps_the_reply() -> None:
    """Each turn sends the system prompt, every earlier turn and the new message, and keeps the reply."""
    model = FakeModel(context_limit=1000, reply_pieces=["Hel", "lo"])
    conversation = new_conversation()
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == ["Hel", "lo"]
    list(conversation.run_turn(model, "Again", no_permission_asked))
    assert model.conversations_received[1] == [SYSTEM_MESSAGE, *exchange("Hi", "Hello"), user("Again")]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "Hello"), *exchange("Again", "Hello")]


def test_turn_keeps_the_tokens_the_model_holds_after_the_reply() -> None:
    """After a turn, the tokens in context are the ones the model counted, not a sum of the text."""
    model = FakeModel(context_limit=1000, reply_pieces=["Hel", "lo"])
    conversation = new_conversation()
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_conversation_below_80_percent_is_sent_whole() -> None:
    """While the conversation is not full, no turn leaves, even when it takes more than half of the context."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=450)
    conversation.add_turn(exchange("q2", "a2"), tokens_in_context_after_reply=800)
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, *exchange("q1", "a1"), *exchange("q2", "a2"), user("Hi")]


def test_full_conversation_is_cut_before_the_message_is_sent() -> None:
    """Above 80% of the context, the oldest turns leave before the new message goes to the model."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = new_conversation()
    # 1000 characters hold 850 tokens, and without the first turn about 433 are left.
    conversation.add_turn(exchange_of_size(491), tokens_in_context_after_reply=450)
    conversation.add_turn(exchange_of_size(500), tokens_in_context_after_reply=850)
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, *exchange_of_size(500), user("Hi")]


def test_conversation_that_does_not_fit_is_sent_again_without_the_earlier_turns() -> None:
    """When the model refuses the conversation, the message goes again with only the system prompt before it."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"])
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == ["Ok"]
    assert model.conversations_received[1] == [SYSTEM_MESSAGE, user("Hi")]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "Ok")]


def test_message_too_long_alone_is_refused_and_the_conversation_stays() -> None:
    """A message that does not fit even alone raises MessageTooLongError and leaves the conversation as it was."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"])
    conversation = new_conversation()
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    messages_before = conversation.messages()
    with pytest.raises(MessageTooLongError):
        list(conversation.run_turn(model, "x" * 200, no_permission_asked))
    assert conversation.messages() == messages_before


def test_retry_that_fails_for_another_reason_leaves_the_conversation_as_it_was() -> None:
    """When the message, sent again without the earlier turns, fails for another reason, those turns come back."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"], failure=ConnectionError("the network is down"))
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    messages_before = conversation.messages()
    with pytest.raises(ConnectionError):
        list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert conversation.messages() == messages_before


def test_reply_closed_after_the_retry_is_kept_without_the_earlier_turns() -> None:
    """A reply closed while the message goes without the earlier turns is kept, and those turns stay out."""
    model = FakeModel(context_limit=100, reply_pieces=["One", " two", " three"])
    conversation = new_conversation()
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    reply = conversation.run_turn(model, "Count", no_permission_asked)
    next(reply)
    reply.close()
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Count", "One")]
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_closed_reply_is_kept_with_what_was_written() -> None:
    """When whoever reads the reply closes it in the middle, the turn stays with the pieces written until then."""
    model = FakeModel(context_limit=1000, reply_pieces=["One", " two", " three"])
    conversation = new_conversation()
    reply = conversation.run_turn(model, "Count", no_permission_asked)
    list(islice(reply, 2))
    reply.close()
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Count", "One two")]
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_message_stays_when_the_reply_ends_before_its_first_piece() -> None:
    """A reply that ends before writing anything, as after a stop() during the prompt, keeps the user's message."""
    model = FakeModel(context_limit=1000, reply_pieces=[])
    conversation = new_conversation()
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == []
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "")]


def test_closing_the_reply_stops_the_model_before_its_tokens_are_counted() -> None:
    """Closing the turn closes the model's reply too, and only then the turn reads the tokens the model holds."""
    model = FakeModel(context_limit=1000, reply_pieces=["One", " two", " three"])
    conversation = new_conversation()
    reply = conversation.run_turn(model, "Count", no_permission_asked)
    next(reply)
    reply.close()
    assert model.events[-2:] == ["reply stopped", "tokens counted"]


def test_conversation_that_does_not_fit_the_real_local_model_is_sent_again_without_the_earlier_turns(
    local_model: LocalModel,
) -> None:
    """With the real model, a conversation over its context is refused and the message goes again alone."""
    conversation = new_conversation()
    huge_question = "Once upon a time " * local_model.context_limit()
    conversation.add_turn(exchange(huge_question, "The end."), tokens_in_context_after_reply=10)
    reply = conversation.run_turn(local_model, "Once upon a time", no_permission_asked)
    list(islice(reply, 3))
    reply.close()
    assert conversation.messages()[:2] == [SYSTEM_MESSAGE, user("Once upon a time")]
    assert len(conversation.messages()) == 3
    assert conversation.tokens_in_context == local_model.tokens_in_use()


def test_message_too_long_for_the_real_local_model_is_refused(local_model: LocalModel) -> None:
    """With the real model, a message over its context raises MessageTooLongError and is not kept."""
    conversation = new_conversation()
    with pytest.raises(MessageTooLongError):
        list(conversation.run_turn(local_model, "Once upon a time " * local_model.context_limit(), no_permission_asked))
    assert conversation.messages() == [SYSTEM_MESSAGE]


def test_turn_on_the_real_local_model_ended_by_stop_keeps_what_was_written(local_model: LocalModel) -> None:
    """With the real model, stop() ends the turn without an error, and the pieces written so far are kept."""
    conversation = new_conversation()
    reply = conversation.run_turn(local_model, "Once upon a time", no_permission_asked)
    first_piece = next(reply)
    local_model.stop()
    assert list(reply) == []
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Once upon a time", first_piece)]


def call(call_id: str, name: str, **arguments: object) -> ToolCall:
    """Return a tool call as a model writes it, with its arguments as JSON."""
    return ToolCall(call_id, name, json.dumps(arguments))


class UserThatAnswers:
    """Stands in for the user: gives the same answer to every question about a tool, and remembers each one."""

    def __init__(self, allows: bool) -> None:
        """Keep the answer, with nothing asked yet."""
        self.allows = allows
        self.asked: list[str] = []

    def __call__(self, use: ToolUse) -> bool:
        """Remember the tool asked about and give the answer."""
        self.asked.append(use.name)
        return self.allows


def test_turn_runs_the_tool_a_reply_asks_for_and_sends_the_result_back(tmp_path: Path) -> None:
    """The result goes to the model in a tool message after the reply that asked, and the model answers again."""
    (tmp_path / "notes.txt").write_text("secret: lighthouse\n")
    model = ScriptedModel(["Looking."], tool_calls_by_reply=[[call("c1", "read_file", path="notes.txt")]])
    conversation = conversation_in(tmp_path)
    events = list(conversation.run_turn(model, "What is the secret?", no_permission_asked))
    assert [type(event) for event in events] == [str, ToolStarted, ToolEnded, str]
    assert events[2].result.text == "secret: lighthouse\n"
    asked = reply_message("Looking.", [call("c1", "read_file", path="notes.txt")])
    result = {"role": "tool", "tool_call_id": "c1", "content": "secret: lighthouse\n"}
    assert model.conversations_received[1][-2:] == [asked, result]
    assert conversation.messages()[1:] == [user("What is the secret?"), asked, result, reply_message("Looking.", [])]


def test_every_reply_of_a_turn_is_offered_the_tools(tmp_path: Path) -> None:
    """Each request of a turn carries the tools of the toolbox."""
    toolbox = Toolbox(tmp_path)
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "read_file", path="missing.txt")]])
    conversation = Conversation(SYSTEM_PROMPT, toolbox, history_in_a_new_folder(), recall=None)
    list(conversation.run_turn(model, "Read it", no_permission_asked))
    assert model.tools_received == [toolbox.schemas(), toolbox.schemas()]


def test_call_that_cannot_run_gives_the_model_the_reason_and_the_turn_goes_on(tmp_path: Path) -> None:
    """A tool that does not exist is answered with an error the model reads, and the model is asked again."""
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "delete_everything")]])
    conversation = conversation_in(tmp_path)
    list(conversation.run_turn(model, "Go", no_permission_asked))
    assert model.conversations_received[1][-1]["content"].startswith("error: there is no tool delete_everything")
    assert len(model.conversations_received) == 2


def test_tool_that_asks_runs_only_after_the_user_allows_it(tmp_path: Path) -> None:
    """A change to a file is asked about first, and done once the user allows it."""
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "write_file", path="new.txt", content="hi")]])
    user_that_allows = UserThatAnswers(allows=True)
    list(conversation_in(tmp_path).run_turn(model, "Write it", user_that_allows))
    assert user_that_allows.asked == ["write_file"]
    assert (tmp_path / "new.txt").read_text() == "hi"


def test_refused_tool_ends_the_turn_and_runs_nothing_after_it(tmp_path: Path) -> None:
    """A refusal changes nothing, leaves the next call of the reply unrun, and asks the model nothing more."""
    calls = [call("c1", "write_file", path="new.txt", content="hi"), call("c2", "run_command", command="touch x")]
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[calls])
    user_that_refuses = UserThatAnswers(allows=False)
    conversation = conversation_in(tmp_path)
    events = list(conversation.run_turn(model, "Do it", user_that_refuses))
    endings = [event.result.ending for event in events if isinstance(event, ToolEnded)]
    assert endings == [Ending.DENIED, Ending.NOT_RUN]
    assert user_that_refuses.asked == ["write_file"]
    assert len(model.conversations_received) == 1
    assert list(tmp_path.iterdir()) == []
    assert [message["role"] for message in conversation.messages()] == ["system", "user", "assistant", "tool", "tool"]


def test_turn_after_a_refusal_runs_as_usual(tmp_path: Path) -> None:
    """The stop a refusal leaves is for its turn only: the next turn asks the model and runs its tools."""
    refused_call = call("c1", "write_file", path="a.txt", content="a")
    allowed_call = call("c2", "write_file", path="b.txt", content="b")
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[refused_call], [allowed_call]])
    conversation = conversation_in(tmp_path)
    list(conversation.run_turn(model, "Write a", UserThatAnswers(allows=False)))
    list(conversation.run_turn(model, "Write b", UserThatAnswers(allows=True)))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["b.txt"]
    assert len(model.conversations_received) == 3


def test_stop_before_a_tool_runs_leaves_it_unrun_and_asks_the_model_nothing_more(tmp_path: Path) -> None:
    """A stop that arrives while a call is shown leaves it unrun, and the turn ends with what was done."""
    (tmp_path / "notes.txt").write_text("hello\n")
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "read_file", path="notes.txt")]])
    conversation = conversation_in(tmp_path)
    events = []
    for event in conversation.run_turn(model, "Read it", no_permission_asked):
        events.append(event)
        if isinstance(event, ToolStarted):
            conversation.stop()
    assert events[-1].result.ending is Ending.NOT_RUN
    assert len(model.conversations_received) == 1
    assert len(conversation.turns) == 1


def test_context_that_fills_after_a_tool_ran_keeps_the_turn_as_cut(tmp_path: Path) -> None:
    """When the result of a tool makes the turn too large, the turn is kept, ended as cut, with no error."""
    (tmp_path / "big.txt").write_text("x" * 500)
    model = ScriptedModel(["Ok."], context_limit=100, tool_calls_by_reply=[[call("c1", "read_file", path="big.txt")]])
    conversation = conversation_in(tmp_path)
    events = list(conversation.run_turn(model, "Read it", no_permission_asked))
    assert isinstance(events[-1], ToolEnded)
    assert conversation.turn_was_cut
    assert [message["role"] for message in conversation.messages()] == ["system", "user", "assistant", "tool"]


def test_turn_closed_while_a_call_is_shown_keeps_every_call_answered(tmp_path: Path) -> None:
    """A turn closed before its calls ran keeps a result for each of them, as the format of the messages needs."""
    calls = [call("c1", "read_file", path="a.txt"), call("c2", "read_file", path="b.txt")]
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[calls])
    conversation = conversation_in(tmp_path)
    turn = conversation.run_turn(model, "Read them", no_permission_asked)
    for event in turn:
        if isinstance(event, ToolStarted):
            break
    turn.close()
    kept = conversation.turns[-1].messages
    assert [(message["role"], message.get("tool_call_id")) for message in kept[-2:]] == [("tool", "c1"), ("tool", "c2")]
    assert kept[-1]["content"] == "Not run: the turn was stopped before this call."
    [saved] = conversation.history.conversations_in_folder()
    [saved_turn] = conversation.history.resume(saved.id)
    assert {call_id: result.ending for call_id, result in saved_turn.tool_results.items()} == {
        "c1": Ending.NOT_RUN,
        "c2": Ending.NOT_RUN,
    }


def test_turn_is_cut_when_its_last_reply_was_cut() -> None:
    """A reply cut by the full context marks its turn as cut, and the next turn starts uncut."""
    model = ScriptedModel(["half a rep"])
    conversation = new_conversation()
    model.cut = True
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert conversation.turn_was_cut
    model.cut = False
    list(conversation.run_turn(model, "Again", no_permission_asked))
    assert not conversation.turn_was_cut


def test_arguments_of_tool_calls_count_as_text_of_the_conversation() -> None:
    """The estimate of the cut counts the arguments a reply wrote for its tools, which also take tokens."""
    asked = reply_message("ab", [ToolCall("c1", "write_file", "x" * 10)])
    assert characters_in([asked, {"role": "tool", "tool_call_id": "c1", "content": "abc"}]) == 15


def test_each_kept_turn_is_saved_with_what_its_tools_gave(tmp_path: Path) -> None:
    """The history gets the turn as the conversation keeps it, and the result of each tool as the screen saw it."""
    (tmp_path / "notes.txt").write_text("hello\n")
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "read_file", path="notes.txt")]])
    conversation = conversation_in(tmp_path)
    list(conversation.run_turn(model, "Read it", no_permission_asked))
    [saved] = conversation.history.conversations_in_folder()
    [saved_turn] = conversation.history.resume(saved.id)
    assert saved_turn.messages == conversation.turns[0].messages
    assert saved_turn.tool_results["c1"].summary == "1 line"
    assert saved.tokens_in_context == conversation.tokens_in_context


def test_message_that_does_not_fit_is_not_saved() -> None:
    """A turn the conversation does not keep leaves nothing in the history either."""
    conversation = new_conversation()
    with pytest.raises(MessageTooLongError):
        list(conversation.run_turn(FakeModel(50, ["Ok"]), "x" * 200, no_permission_asked))
    assert conversation.history.conversations_in_folder() == []


def test_conversation_continued_sends_its_earlier_turns_and_cuts_by_its_saved_count() -> None:
    """The turns read back go to the model with the next message, and the saved count decides the cut."""
    conversation = new_conversation()
    conversation.continue_from([exchange("Hi", "Hello."), exchange("More", "Sure.")], tokens_in_context=10)
    model = FakeModel(1000, ["Ok"])
    list(conversation.run_turn(model, "Again", no_permission_asked))
    assert model.conversations_received[0][1:] == exchange("Hi", "Hello.") + exchange("More", "Sure.") + [user("Again")]
    conversation.continue_from([exchange_of_size(500), exchange_of_size(500)], tokens_in_context=900)
    assert conversation.is_full(1000)


class RecallThatFinds:
    """Stands in for the search of earlier turns: it brings back the same excerpts each time, or fails as told."""

    def __init__(self, excerpts: list[str], failure: Exception | None = None) -> None:
        """Keep what each search gives, with no search made yet and nothing done during one."""
        self.excerpts = excerpts
        self.failure = failure
        self.searches: list[tuple[str, int]] = []
        self.stops_given: list[threading.Event] = []
        self.while_searching: Callable[[], None] = lambda: None

    def closest_excerpts(self, user_text: str, turns_in_context: int, stop_requested: threading.Event) -> list[str]:
        """Remember what the search was asked for, do what the test set for its middle, and give its excerpts."""
        self.searches.append((user_text, turns_in_context))
        self.stops_given.append(stop_requested)
        self.while_searching()
        if self.failure is not None:
            raise self.failure
        return self.excerpts


def conversation_recalling(recall: RecallThatFinds, folder: Path = Path("no-tool-runs-here")) -> Conversation:
    """Return a conversation that searches earlier turns with the stand-in, whose tools act from the folder."""
    return Conversation(SYSTEM_PROMPT, Toolbox(folder), history_in_a_new_folder(), recall)


def test_model_reads_the_recalled_turns_with_the_message_but_the_turn_keeps_the_message_alone() -> None:
    """What was recalled goes only to the model: the conversation and the history keep what the user wrote."""
    excerpt = "User: a\nAssistant: b"
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds([excerpt]))
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == [Recalled(1), "Ok"]
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, with_recalled(user("Hi"), [excerpt])]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "Ok")]
    [saved] = conversation.history.conversations_in_folder()
    assert conversation.history.resume(saved.id)[0].messages == exchange("Hi", "Ok")


def test_recalled_turns_go_with_every_reply_of_the_turn_and_not_with_the_next_turn(tmp_path: Path) -> None:
    """The reply after a tool reads the same recalled turns; the next turn reads only its own search."""
    (tmp_path / "notes.txt").write_text("hello\n")
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[call("c1", "read_file", path="notes.txt")]])
    recall = RecallThatFinds(["User: a\nAssistant: b"])
    conversation = conversation_recalling(recall, tmp_path)
    list(conversation.run_turn(model, "Read it", no_permission_asked))
    first_reply, reply_after_the_tool = model.conversations_received
    assert first_reply[1] == reply_after_the_tool[1] == with_recalled(user("Read it"), ["User: a\nAssistant: b"])
    recall.excerpts = []
    list(conversation.run_turn(model, "Thanks", no_permission_asked))
    assert model.conversations_received[2][1] == user("Read it")
    assert model.conversations_received[2][-1] == user("Thanks")


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("the connection to OpenRouter failed: no route to host"),
        sqlite3.OperationalError("database is locked"),
        KeyError("data"),
    ],
)
def test_search_that_fails_is_told_and_the_turn_goes_on_without_it(failure: Exception) -> None:
    """A failed search, of the network, of the history or of the answer, ends nothing: the model gets the message."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds([], failure))
    events = list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert events == [RecallFailed(str(failure)), "Ok"]
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, user("Hi")]


def test_search_is_given_the_stop_of_the_conversation() -> None:
    """The search sees the stop of the turn, so a long indexing ends when the user stops the turn."""
    recall = RecallThatFinds([])
    conversation = conversation_recalling(recall)
    list(conversation.run_turn(FakeModel(context_limit=1000, reply_pieces=["Ok"]), "Hi", no_permission_asked))
    assert recall.stops_given == [conversation.stop_requested]


def test_search_that_fails_after_one_that_recalled_sends_none_of_the_turns_recalled_before() -> None:
    """What an earlier search brought back is for its own turn: a failed search leaves the message alone."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    recall = RecallThatFinds(["User: a\nAssistant: b"])
    conversation = conversation_recalling(recall)
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    recall.failure = ConnectionError("the connection to OpenRouter failed: no route to host")
    list(conversation.run_turn(model, "Again", no_permission_asked))
    assert model.conversations_received[1][-1] == user("Again")


def test_search_that_finds_nothing_tells_nothing() -> None:
    """With no turn close enough, the turn runs as it would with no search."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds([]))
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == ["Ok"]
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, user("Hi")]


def test_search_runs_after_the_cut_and_leaves_out_the_turns_the_model_still_reads() -> None:
    """The search gets how many turns stay in the context once the cut ran, so the turns cut can be recalled."""
    recall = RecallThatFinds([])
    conversation = conversation_recalling(recall)
    for _ in range(3):
        conversation.add_turn(exchange_of_size(300), tokens_in_context_after_reply=900)
    list(conversation.run_turn(FakeModel(context_limit=1000, reply_pieces=["Ok"]), "Hi", no_permission_asked))
    assert recall.searches == [("Hi", 1)]


def test_turn_stopped_during_the_search_asks_the_model_nothing(tmp_path: Path) -> None:
    """A stop during the search keeps the message with an empty reply, and nothing of the last reply is taken again."""
    write = call("c1", "write_file", path="new.txt", content="hi")
    model = ScriptedModel(["Ok."], tool_calls_by_reply=[[write]])
    recall = RecallThatFinds([])
    conversation = conversation_recalling(recall, tmp_path)
    list(conversation.run_turn(model, "Write it", UserThatAnswers(allows=False)))
    model.cut = True
    recall.while_searching = conversation.stop
    assert list(conversation.run_turn(model, "Again", UserThatAnswers(allows=True))) == []
    assert conversation.turns[-1].messages == exchange("Again", "")
    assert not conversation.turn_was_cut
    assert len(model.conversations_received) == 1
    assert list(tmp_path.iterdir()) == []


def test_recalled_turns_leave_before_the_earlier_turns_when_the_conversation_does_not_fit() -> None:
    """What was recalled is only an aid: when it is what overflows, it leaves and the earlier turns stay."""
    model = FakeModel(context_limit=300, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds(["User: a\nAssistant: " + "b" * 200]))
    conversation.add_turn(exchange_of_size(100), tokens_in_context_after_reply=10)
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == [Recalled(1), "Ok"]
    assert model.conversations_received[1] == [SYSTEM_MESSAGE, *exchange_of_size(100), user("Hi")]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange_of_size(100), *exchange("Hi", "Ok")]


def test_tokens_kept_after_a_recall_leave_out_what_was_recalled() -> None:
    """The count kept for the cut is of what the next turn sends, which no longer carries the recalled turns."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds(["User: a\nAssistant: " + "b" * 300]))
    list(conversation.run_turn(model, "Hi", no_permission_asked))
    assert model.tokens > characters_in(conversation.messages())
    assert conversation.tokens_in_context == characters_in(conversation.messages())
    [saved] = conversation.history.conversations_in_folder()
    assert saved.tokens_in_context == conversation.tokens_in_context


def test_message_sent_again_without_the_earlier_turns_goes_without_what_was_recalled() -> None:
    """When even the conversation without the recalled turns does not fit, the earlier turns leave too."""
    model = FakeModel(context_limit=50, reply_pieces=["Ok"])
    conversation = conversation_recalling(RecallThatFinds(["User: a\nAssistant: " + "b" * 100]))
    conversation.add_turn(exchange_of_size(100), tokens_in_context_after_reply=10)
    assert list(conversation.run_turn(model, "Hi", no_permission_asked)) == [Recalled(1), "Ok"]
    with_the_earlier_turns, alone = model.conversations_received[1:]
    assert with_the_earlier_turns == [SYSTEM_MESSAGE, *exchange_of_size(100), user("Hi")]
    assert alone == [SYSTEM_MESSAGE, user("Hi")]
