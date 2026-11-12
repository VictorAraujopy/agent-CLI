"""Tests for the conversation: the messages it keeps, the cut of the oldest turns, and each turn it runs."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pytest

from agent_cli.agent.conversation import Conversation, MessageTooLongError
from agent_cli.inference.generation import LocalModel
from agent_cli.inference.loader import load_model
from agent_cli.inference.model import ContextFullError, Model

SYSTEM_PROMPT = "Be brief."
SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}
PIECES_TO_READ = 3


@pytest.fixture(scope="module")
def local_model(small_model_file: Path) -> LocalModel:
    """Return the small real model, loaded once for the tests that run the conversation on it."""
    return LocalModel(load_model(small_model_file))


class FakeModel(Model):
    """A model for the tests: its tokens are characters, it refuses what passes its limit, and its reply is fixed."""

    def __init__(self, context_limit: int, reply_pieces: list[str], failure: Exception | None = None) -> None:
        """Keep the limit, the pieces every reply will have, and an error to raise instead of them, if any."""
        self.limit = context_limit
        self.reply_pieces = reply_pieces
        self.failure = failure
        self.conversations_received: list[list[dict[str, str]]] = []
        self.tokens = 0
        self.events: list[str] = []

    def reply(self, messages: list[dict[str, str]]) -> Iterator[str]:
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

    def tokens_in_use(self) -> int:
        """Return the characters of the last conversation and of the reply written so far."""
        self.events.append("tokens counted")
        return self.tokens

    def stop(self) -> None:
        """Do nothing: the tests end the fake's reply by closing it."""

    def reply_was_cut(self) -> bool:
        """Tell that the last reply was not cut: the fake always finishes its reply."""
        return False

    def context_limit(self) -> int:
        """Return the limit given when the model was made."""
        return self.limit


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
    conversation = Conversation(SYSTEM_PROMPT)
    for size in sizes:
        conversation.add_turn(exchange_of_size(size), tokens_in_context)
    return conversation


def test_new_conversation_holds_only_the_system_prompt() -> None:
    """Before any turn, the only message is the system prompt, and no token is counted."""
    conversation = Conversation(SYSTEM_PROMPT)
    assert conversation.messages() == [SYSTEM_MESSAGE]
    assert conversation.tokens_in_context == 0


def test_tokens_in_context_are_the_last_count_of_the_model() -> None:
    """After each turn, the tokens in context are what the model counted then, even when the count went down."""
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=900)
    conversation.add_turn(exchange("q2", "a2"), tokens_in_context_after_reply=500)
    assert conversation.tokens_in_context == 500


def test_conversation_is_full_only_above_80_percent_of_the_context() -> None:
    """At exactly 80% of the context the conversation is not full yet; one token more and it is."""
    conversation = Conversation(SYSTEM_PROMPT)
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
    conversation = Conversation("s" * 400)
    conversation.add_turn(exchange_of_size(300), tokens_in_context_after_reply=700)
    conversation.add_turn(exchange_of_size(300), tokens_in_context_after_reply=1000)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.turns == []
    assert conversation.tokens_in_context == 400


def test_cut_depends_only_on_the_last_count_of_the_model() -> None:
    """Earlier counts change nothing, not even one that went down, as when a model drops its old reasoning."""
    count_went_down = Conversation(SYSTEM_PROMPT)
    count_went_down.add_turn(exchange_of_size(491), tokens_in_context_after_reply=950)
    count_went_down.add_turn(exchange_of_size(500), tokens_in_context_after_reply=900)
    count_went_up = Conversation(SYSTEM_PROMPT)
    count_went_up.add_turn(exchange_of_size(491), tokens_in_context_after_reply=100)
    count_went_up.add_turn(exchange_of_size(500), tokens_in_context_after_reply=900)
    count_went_down.cut_oldest_turns(context_limit=1000)
    count_went_up.cut_oldest_turns(context_limit=1000)
    assert count_went_down.messages() == count_went_up.messages()


def test_cut_never_drops_the_system_prompt() -> None:
    """Even when every turn has to go, the system prompt stays."""
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=5000)
    conversation.cut_oldest_turns(context_limit=1000)
    assert conversation.messages() == [SYSTEM_MESSAGE]


def test_turn_sends_the_whole_conversation_and_keeps_the_reply() -> None:
    """Each turn sends the system prompt, every earlier turn and the new message, and keeps the reply."""
    model = FakeModel(context_limit=1000, reply_pieces=["Hel", "lo"])
    conversation = Conversation(SYSTEM_PROMPT)
    assert list(conversation.run_turn(model, "Hi")) == ["Hel", "lo"]
    list(conversation.run_turn(model, "Again"))
    assert model.conversations_received[1] == [SYSTEM_MESSAGE, *exchange("Hi", "Hello"), user("Again")]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "Hello"), *exchange("Again", "Hello")]


def test_turn_keeps_the_tokens_the_model_holds_after_the_reply() -> None:
    """After a turn, the tokens in context are the ones the model counted, not a sum of the text."""
    model = FakeModel(context_limit=1000, reply_pieces=["Hel", "lo"])
    conversation = Conversation(SYSTEM_PROMPT)
    list(conversation.run_turn(model, "Hi"))
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_conversation_below_80_percent_is_sent_whole() -> None:
    """While the conversation is not full, no turn leaves, even when it takes more than half of the context."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "a1"), tokens_in_context_after_reply=450)
    conversation.add_turn(exchange("q2", "a2"), tokens_in_context_after_reply=800)
    list(conversation.run_turn(model, "Hi"))
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, *exchange("q1", "a1"), *exchange("q2", "a2"), user("Hi")]


def test_full_conversation_is_cut_before_the_message_is_sent() -> None:
    """Above 80% of the context, the oldest turns leave before the new message goes to the model."""
    model = FakeModel(context_limit=1000, reply_pieces=["Ok"])
    conversation = Conversation(SYSTEM_PROMPT)
    # 1000 characters hold 850 tokens, and without the first turn about 433 are left.
    conversation.add_turn(exchange_of_size(491), tokens_in_context_after_reply=450)
    conversation.add_turn(exchange_of_size(500), tokens_in_context_after_reply=850)
    list(conversation.run_turn(model, "Hi"))
    assert model.conversations_received[0] == [SYSTEM_MESSAGE, *exchange_of_size(500), user("Hi")]


def test_conversation_that_does_not_fit_is_sent_again_without_the_earlier_turns() -> None:
    """When the model refuses the conversation, the message goes again with only the system prompt before it."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"])
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    assert list(conversation.run_turn(model, "Hi")) == ["Ok"]
    assert model.conversations_received[1] == [SYSTEM_MESSAGE, user("Hi")]
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "Ok")]


def test_message_too_long_alone_is_refused_and_the_conversation_stays() -> None:
    """A message that does not fit even alone raises MessageTooLongError and leaves the conversation as it was."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"])
    conversation = Conversation(SYSTEM_PROMPT)
    list(conversation.run_turn(model, "Hi"))
    messages_before = conversation.messages()
    with pytest.raises(MessageTooLongError):
        list(conversation.run_turn(model, "x" * 200))
    assert conversation.messages() == messages_before


def test_retry_that_fails_for_another_reason_leaves_the_conversation_as_it_was() -> None:
    """When the message, sent again without the earlier turns, fails for another reason, those turns come back."""
    model = FakeModel(context_limit=100, reply_pieces=["Ok"], failure=ConnectionError("the network is down"))
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    messages_before = conversation.messages()
    with pytest.raises(ConnectionError):
        list(conversation.run_turn(model, "Hi"))
    assert conversation.messages() == messages_before


def test_reply_closed_after_the_retry_is_kept_without_the_earlier_turns() -> None:
    """A reply closed while the message goes without the earlier turns is kept, and those turns stay out."""
    model = FakeModel(context_limit=100, reply_pieces=["One", " two", " three"])
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_turn(exchange("q1", "x" * 100), tokens_in_context_after_reply=10)
    reply = conversation.run_turn(model, "Count")
    next(reply)
    reply.close()
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Count", "One")]
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_closed_reply_is_kept_with_what_was_written() -> None:
    """When whoever reads the reply closes it in the middle, the turn stays with the pieces written until then."""
    model = FakeModel(context_limit=1000, reply_pieces=["One", " two", " three"])
    conversation = Conversation(SYSTEM_PROMPT)
    reply = conversation.run_turn(model, "Count")
    list(islice(reply, 2))
    reply.close()
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Count", "One two")]
    assert conversation.tokens_in_context == model.tokens_in_use()


def test_message_stays_when_the_reply_ends_before_its_first_piece() -> None:
    """A reply that ends before writing anything, as after a stop() during the prompt, keeps the user's message."""
    model = FakeModel(context_limit=1000, reply_pieces=[])
    conversation = Conversation(SYSTEM_PROMPT)
    assert list(conversation.run_turn(model, "Hi")) == []
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Hi", "")]


def test_closing_the_reply_stops_the_model_before_its_tokens_are_counted() -> None:
    """Closing the turn closes the model's reply too, and only then the turn reads the tokens the model holds."""
    model = FakeModel(context_limit=1000, reply_pieces=["One", " two", " three"])
    conversation = Conversation(SYSTEM_PROMPT)
    reply = conversation.run_turn(model, "Count")
    next(reply)
    reply.close()
    assert model.events[-2:] == ["reply stopped", "tokens counted"]


def test_conversation_that_does_not_fit_the_real_local_model_is_sent_again_without_the_earlier_turns(
    local_model: LocalModel,
) -> None:
    """With the real model, a conversation over its context is refused and the message goes again alone."""
    conversation = Conversation(SYSTEM_PROMPT)
    huge_question = "Once upon a time " * local_model.context_limit()
    conversation.add_turn(exchange(huge_question, "The end."), tokens_in_context_after_reply=10)
    reply = conversation.run_turn(local_model, "Once upon a time")
    list(islice(reply, PIECES_TO_READ))
    reply.close()
    assert conversation.messages()[:2] == [SYSTEM_MESSAGE, user("Once upon a time")]
    assert len(conversation.messages()) == 3
    assert conversation.tokens_in_context == local_model.tokens_in_use()


def test_message_too_long_for_the_real_local_model_is_refused(local_model: LocalModel) -> None:
    """With the real model, a message over its context raises MessageTooLongError and is not kept."""
    conversation = Conversation(SYSTEM_PROMPT)
    with pytest.raises(MessageTooLongError):
        list(conversation.run_turn(local_model, "Once upon a time " * local_model.context_limit()))
    assert conversation.messages() == [SYSTEM_MESSAGE]


def test_turn_on_the_real_local_model_ended_by_stop_keeps_what_was_written(local_model: LocalModel) -> None:
    """With the real model, stop() ends the turn without an error, and the pieces written so far are kept."""
    conversation = Conversation(SYSTEM_PROMPT)
    reply = conversation.run_turn(local_model, "Once upon a time")
    first_piece = next(reply)
    local_model.stop()
    assert list(reply) == []
    assert conversation.messages() == [SYSTEM_MESSAGE, *exchange("Once upon a time", first_piece)]
