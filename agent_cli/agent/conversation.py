"""The conversation: keeps the messages, cuts the oldest turns when the context fills, and runs each turn."""

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from agent_cli.history.queries import History
from agent_cli.inference.model import ContextFullError, Message, Model, ToolCall
from agent_cli.rag.recall import Recall, with_recalled
from agent_cli.tools.tool import Ending, ToolResult
from agent_cli.tools.toolbox import Toolbox, ToolUse

NOT_RUN = ToolResult("Not run: the turn was stopped before this call.", "not run", Ending.NOT_RUN)


class MessageTooLongError(Exception):
    """The user's message does not fit in the model's context, even with every earlier turn left out."""


@dataclass
class Turn:
    """One exchange: the user's message, the model's replies and the results of the tools they asked for."""

    messages: list[Message]


@dataclass(frozen=True)
class ToolStarted:
    """A reply asked for a tool: the call as the user sees it, before it is allowed and run."""

    use: ToolUse


@dataclass(frozen=True)
class ToolEnded:
    """A tool call ended, run or not, with the result the model reads next."""

    use: ToolUse
    result: ToolResult


@dataclass(frozen=True)
class Recalled:
    """The search brought back this many earlier turns, which the model reads with the user's message."""

    turns: int


@dataclass(frozen=True)
class RecallFailed:
    """The search of earlier turns failed, for this reason, and the turn goes on without it."""

    reason: str


# What a turn tells whoever runs it, in order: what the search found, each piece of a reply, each tool.
TurnEvent = Recalled | RecallFailed | str | ToolStarted | ToolEnded


def characters_in(messages: list[Message]) -> int:
    """Return how many characters of text the messages hold, the arguments of their tool calls included."""
    characters = 0
    for message in messages:
        characters += len(message["content"])
        for call in message.get("tool_calls", []):
            characters += len(call["function"]["arguments"])
    return characters


def reply_message(text: str, calls: list[ToolCall]) -> Message:
    """Return the message of a model's reply, with the tools it asked for, in the chat completions format."""
    message: Message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ]
    return message


def tool_message(call_id: str, result_text: str) -> Message:
    """Return the message that gives the model the result of one of its tool calls."""
    return {"role": "tool", "tool_call_id": call_id, "content": result_text}


def calls_with_no_result(messages: list[Message]) -> list[str]:
    """Return the ids of the tool calls in the messages that no tool message answers, in order."""
    answered = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
    asked: list[str] = []
    for message in messages:
        asked += [call["id"] for call in message.get("tool_calls", [])]
    return [call_id for call_id in asked if call_id not in answered]


class Conversation:
    """What is sent to the model: the system prompt first, then the turns that still fit in the context."""

    def __init__(self, system_prompt: str, toolbox: Toolbox, history: History, recall: Recall | None) -> None:
        """Start a conversation with only the system prompt, which offers the tools, saves each turn and may recall."""
        self.system_message = {"role": "system", "content": system_prompt}
        self.toolbox = toolbox
        self.history = history
        # None when no embedding model is configured: then the turns never leave this machine to be searched.
        self.recall = recall
        # What the search brought back for the turn being run, which the model reads with the user's message.
        self.recalled: list[str] = []
        self.turns: list[Turn] = []
        # The tokens the model counted in its context when the last turn ended. After a cut it is an estimate,
        # until the next turn ends and the model counts again.
        self.tokens_in_context = 0
        # Set by stop(): the turn being run asks the model for nothing more and runs no more tools.
        self.stop_requested = threading.Event()
        # Whether the last turn ended because the context filled, in the middle of a reply or between two.
        self.turn_was_cut = False
        # What each tool call of the turn being run gave, by the id of the call, for the history to keep.
        self.tool_results: dict[str, ToolResult] = {}

    def messages(self) -> list[Message]:
        """Return every message to send, in order: the system prompt, then the messages of each turn."""
        messages = [self.system_message]
        for turn in self.turns:
            messages.extend(turn.messages)
        return messages

    def add_turn(self, messages: list[Message], tokens_in_context_after_reply: int) -> None:
        """Keep a finished turn, and the tokens the model counted in its context when the turn ended."""
        self.turns.append(Turn(messages))
        self.tokens_in_context = tokens_in_context_after_reply

    def continue_from(self, turns: list[list[Message]], tokens_in_context: int) -> None:
        """Take the turns of a conversation read back from the history, and the tokens it took when it was left."""
        self.turns = [Turn(messages) for messages in turns]
        self.tokens_in_context = tokens_in_context

    def is_full(self, context_limit: int) -> bool:
        """Tell whether the conversation passed the share of the context where the cut starts."""
        return self.tokens_in_context > context_limit * 0.8

    def cut_oldest_turns(self, context_limit: int) -> None:
        """Drop the oldest turns until the conversation takes at most the share kept after a cut."""
        tokens_to_keep = context_limit * 0.5
        # The model only counts the whole context, so each message is taken to hold the share of the tokens
        # that matches the share of the text it holds.
        tokens_per_character = self.tokens_in_context / characters_in(self.messages())
        while self.turns and self.tokens_in_context > tokens_to_keep:
            self.turns.pop(0)
            self.tokens_in_context = round(characters_in(self.messages()) * tokens_per_character)

    def stop(self) -> None:
        """Ask the turn being run to end: no more tools run and the model is not asked again; safe from any thread."""
        self.stop_requested.set()

    def run_turn(
        self, model: Model, user_text: str, ask_permission: Callable[[ToolUse], bool]
    ) -> Iterator[TurnEvent]:
        """Send the user's message with what the search recalls, yield each reply and each tool, and keep it all."""
        self.stop_requested.clear()
        self.turn_was_cut = False
        self.tool_results = {}
        context_limit = model.context_limit()
        if self.is_full(context_limit):
            self.cut_oldest_turns(context_limit)
        # After the cut, so the turns it just left out can be recalled.
        yield from self.search_earlier_turns(user_text)
        turn_messages: list[Message] = [{"role": "user", "content": user_text}]
        try:
            yield from self.replies_and_tools(model, turn_messages, ask_permission)
        except GeneratorExit:
            # Whoever reads the turn closed it before the end: what was written and run so far stays.
            self.keep_turn(model, turn_messages)
            raise
        self.keep_turn(model, turn_messages)

    def search_earlier_turns(self, user_text: str) -> Iterator[Recalled | RecallFailed]:
        """Keep the earlier turns closest to the message for the model to read, and yield how the search went."""
        self.recalled = []
        if self.recall is None:
            return
        try:
            self.recalled = self.recall.closest_excerpts(user_text, len(self.turns), self.stop_requested)
        except Exception as error:
            # The search only adds to the turn, so no failure of it, of the network or of the history, ends the turn.
            yield RecallFailed(str(error))
            return
        if self.recalled:
            yield Recalled(len(self.recalled))

    def messages_to_send(self, turn_messages: list[Message]) -> list[Message]:
        """Return what the model reads: the conversation, then the turn, its first message with what was recalled."""
        user_message, *rest_of_the_turn = turn_messages
        return self.messages() + [with_recalled(user_message, self.recalled), *rest_of_the_turn]

    def replies_and_tools(
        self, model: Model, turn_messages: list[Message], ask_permission: Callable[[ToolUse], bool]
    ) -> Iterator[str | ToolStarted | ToolEnded]:
        """Yield each reply and the tools it asks for, until a reply asks for none or the turn stops."""
        if self.stop_requested.is_set():
            # Stopped during the search: the model is not asked, and the reply stays empty, as one stopped at once.
            turn_messages.append(reply_message("", []))
            return
        while True:
            try:
                yield from self.next_reply(model, turn_messages)
            except MessageTooLongError:
                nothing_ran_yet = len(turn_messages) == 1
                if nothing_ran_yet:
                    raise
                # Tools already ran, so the turn is kept, ended as a reply cut by the full context.
                self.turn_was_cut = True
                return
            calls = model.tool_calls()
            if not calls:
                return
            yield from self.run_tools(calls, turn_messages, ask_permission)
            if self.stop_requested.is_set():
                return

    def next_reply(self, model: Model, turn_messages: list[Message]) -> Iterator[str]:
        """Yield the model's next reply piece by piece, and keep it in the turn with the tools it asks for."""
        reply_pieces: list[str] = []
        pieces = self.reply_that_fits(model, turn_messages)
        try:
            for piece in pieces:
                reply_pieces.append(piece)
                yield piece
        except GeneratorExit:
            # The model stops too, and what was shown stays.
            pieces.close()
            turn_messages.append(reply_message("".join(reply_pieces), []))
            raise
        turn_messages.append(reply_message("".join(reply_pieces), model.tool_calls()))
        self.turn_was_cut = self.turn_was_cut or model.reply_was_cut()

    def reply_that_fits(self, model: Model, turn_messages: list[Message]) -> Iterator[str]:
        """Yield the reply to the conversation and the turn, leaving out what was recalled first if they do not fit."""
        try:
            yield from model.reply(self.messages_to_send(turn_messages), self.toolbox.schemas())
        except ContextFullError:
            yield from self.reply_without_what_was_recalled(model, turn_messages)

    def reply_without_what_was_recalled(self, model: Model, turn_messages: list[Message]) -> Iterator[str]:
        """Yield the reply with no recalled turns; with none to leave out, or still too long, with no earlier turns."""
        if not self.recalled:
            yield from self.reply_with_no_earlier_turns(model, turn_messages)
            return
        self.recalled = []
        try:
            yield from model.reply(self.messages_to_send(turn_messages), self.toolbox.schemas())
        except ContextFullError:
            yield from self.reply_with_no_earlier_turns(model, turn_messages)

    def reply_with_no_earlier_turns(self, model: Model, turn_messages: list[Message]) -> Iterator[str]:
        """Yield the reply to the system prompt and the turn alone, dropping every earlier turn."""
        earlier_turns = self.turns
        self.turns = []
        try:
            yield from model.reply(self.messages_to_send(turn_messages), self.toolbox.schemas())
        except ContextFullError as error:
            # Not even the turn alone fits. It is not sent, so the conversation goes back to how it was.
            self.turns = earlier_turns
            raise MessageTooLongError(str(error)) from error
        except Exception:
            # Any other failure keeps no turn, so the earlier turns come back too. Closing the reply is not an
            # Exception, and stop() ends it with no error: both keep the turn without the earlier turns, as sent.
            self.turns = earlier_turns
            raise

    def run_tools(
        self, calls: list[ToolCall], turn_messages: list[Message], ask_permission: Callable[[ToolUse], bool]
    ) -> Iterator[ToolStarted | ToolEnded]:
        """Run each call the user allows, yielding it before and after, and keep each result in the turn."""
        for call in calls:
            use = self.toolbox.read(call.name, call.arguments)
            yield ToolStarted(use)
            result = self.result_of(use, ask_permission)
            self.tool_results[call.id] = result
            turn_messages.append(tool_message(call.id, result.text))
            yield ToolEnded(use, result)

    def result_of(self, use: ToolUse, ask_permission: Callable[[ToolUse], bool]) -> ToolResult:
        """Run the call if the turn goes on and the user allows it; a refusal stops the turn."""
        if self.stop_requested.is_set():
            return NOT_RUN
        if use.asks_permission and not ask_permission(use):
            self.stop_requested.set()
            denial = "The user did not allow this call. Do not try it again: wait for the user's next message."
            return ToolResult(denial, "denied", Ending.DENIED)
        return self.toolbox.run(use, self.stop_requested)

    def keep_turn(self, model: Model, turn_messages: list[Message]) -> None:
        """Keep the messages of the turn, every tool call with a result, in the history and in the conversation."""
        for call_id in calls_with_no_result(turn_messages):
            turn_messages.append(tool_message(call_id, NOT_RUN.text))
            self.tool_results[call_id] = NOT_RUN
        tokens_in_context = self.tokens_without_what_was_recalled(model.tokens_in_use(), turn_messages)
        self.history.save_turn(turn_messages, self.tool_results, tokens_in_context)
        self.add_turn(turn_messages, tokens_in_context)

    def tokens_without_what_was_recalled(self, tokens_counted: int, turn_messages: list[Message]) -> int:
        """Return the tokens the model counted less the share of what was recalled, which no later turn sends."""
        characters_sent = characters_in(self.messages_to_send(turn_messages))
        characters_kept = characters_in(self.messages() + turn_messages)
        return round(tokens_counted * characters_kept / characters_sent)
