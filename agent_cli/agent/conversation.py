"""The conversation: keeps the messages, cuts the oldest turns when the context fills, and runs each turn."""

from collections.abc import Iterator
from dataclasses import dataclass

from agent_cli.inference.model import ContextFullError, Model

# Shares of the model's context, from 0 to 1.
CUT_WHEN_ABOVE = 0.8
CUT_DOWN_TO = 0.5


class MessageTooLongError(Exception):
    """The user's message does not fit in the model's context, even with every earlier turn left out."""


@dataclass
class Turn:
    """One exchange: the user's message and what answered it."""

    messages: list[dict[str, str]]


def characters_in(messages: list[dict[str, str]]) -> int:
    """Return how many characters of text the messages hold."""
    return sum(len(message["content"]) for message in messages)


class Conversation:
    """What is sent to the model: the system prompt first, then the turns that still fit in the context."""

    def __init__(self, system_prompt: str) -> None:
        """Start a conversation that holds only the system prompt, with no tokens counted yet."""
        self.system_message = {"role": "system", "content": system_prompt}
        self.turns: list[Turn] = []
        # The tokens the model counted in its context when the last turn ended. After a cut it is an estimate,
        # until the next turn ends and the model counts again.
        self.tokens_in_context = 0

    def messages(self) -> list[dict[str, str]]:
        """Return every message to send, in order: the system prompt, then the messages of each turn."""
        messages = [self.system_message]
        for turn in self.turns:
            messages.extend(turn.messages)
        return messages

    def add_turn(self, messages: list[dict[str, str]], tokens_in_context_after_reply: int) -> None:
        """Keep a finished turn, and the tokens the model counted in its context when the turn ended."""
        self.turns.append(Turn(messages))
        self.tokens_in_context = tokens_in_context_after_reply

    def is_full(self, context_limit: int) -> bool:
        """Tell whether the conversation passed the share of the context where the cut starts."""
        return self.tokens_in_context > context_limit * CUT_WHEN_ABOVE

    def cut_oldest_turns(self, context_limit: int) -> None:
        """Drop the oldest turns until the conversation takes at most the share kept after a cut."""
        tokens_to_keep = context_limit * CUT_DOWN_TO
        # The model only counts the whole context, so each message is taken to hold the share of the tokens
        # that matches the share of the text it holds.
        tokens_per_character = self.tokens_in_context / characters_in(self.messages())
        while self.turns and self.tokens_in_context > tokens_to_keep:
            self.turns.pop(0)
            self.tokens_in_context = round(characters_in(self.messages()) * tokens_per_character)

    def run_turn(self, model: Model, user_text: str) -> Iterator[str]:
        """Send the user's message with the conversation, yield the reply piece by piece, and keep both."""
        context_limit = model.context_limit()
        if self.is_full(context_limit):
            self.cut_oldest_turns(context_limit)
        user_message = {"role": "user", "content": user_text}
        reply_pieces: list[str] = []
        pieces = self.reply_that_fits(model, user_message)
        try:
            for piece in pieces:
                reply_pieces.append(piece)
                yield piece
        except GeneratorExit:
            # Whoever reads the reply closed it before the end. The model stops too, and what was shown stays.
            pieces.close()
            self.keep_turn(model, user_message, reply_pieces)
            raise
        self.keep_turn(model, user_message, reply_pieces)

    def reply_that_fits(self, model: Model, user_message: dict[str, str]) -> Iterator[str]:
        """Yield the reply to the conversation and the new message, leaving the earlier turns out if they do not fit."""
        try:
            yield from model.reply(self.messages() + [user_message])
        except ContextFullError:
            yield from self.reply_with_no_earlier_turns(model, user_message)

    def reply_with_no_earlier_turns(self, model: Model, user_message: dict[str, str]) -> Iterator[str]:
        """Yield the reply to the system prompt and the new message alone, dropping every earlier turn."""
        earlier_turns = self.turns
        self.turns = []
        try:
            yield from model.reply(self.messages() + [user_message])
        except ContextFullError as error:
            # Not even the message alone fits. It is not sent, so the conversation goes back to how it was.
            self.turns = earlier_turns
            raise MessageTooLongError(str(error)) from error
        except Exception:
            # Any other failure keeps no turn, so the earlier turns come back too. Closing the reply is not an
            # Exception, and stop() ends it with no error: both keep the turn without the earlier turns, as sent.
            self.turns = earlier_turns
            raise

    def keep_turn(self, model: Model, user_message: dict[str, str], reply_pieces: list[str]) -> None:
        """Keep the user's message and the reply written to it, with the tokens the model holds now."""
        reply_message = {"role": "assistant", "content": "".join(reply_pieces)}
        self.add_turn([user_message, reply_message], model.tokens_in_use())
