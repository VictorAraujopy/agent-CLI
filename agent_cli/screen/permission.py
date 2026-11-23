"""Asking the user whether a tool may run: the answer the thread of the turn waits for, and the line that asks."""

import threading
from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Static


class PermissionAnswer:
    """The user's answer about one tool call, given on the screen and waited for by the thread of the turn."""

    def __init__(self) -> None:
        """Start with no answer given."""
        self.given = threading.Event()
        self.allowed = False

    def give(self, allowed: bool) -> None:
        """Give the answer, which ends the wait of the thread; safe to call from any thread."""
        self.allowed = allowed
        self.given.set()

    def wait(self, stop_requested: threading.Event) -> bool:
        """Wait for the answer; a turn stopped meanwhile, as when the screen closes, counts as a refusal."""
        while not self.given.wait(timeout=0.1):
            if stop_requested.is_set():
                return False
        return self.allowed


class PermissionPrompt(Static):
    """The line under a tool that asks whether it may run, answered with y or n."""

    can_focus = True
    BINDINGS = [("y", "answer(True)", "Allow"), ("n", "answer(False)", "Refuse")]

    @dataclass
    class Answered(Message):
        """The user answered the prompt: allowed or not."""

        prompt: "PermissionPrompt"
        allowed: bool

    def __init__(self, answer: PermissionAnswer) -> None:
        """Keep the answer the thread of the turn waits for."""
        super().__init__("allow?  y yes   n no   esc stops the turn", markup=False)
        self.answer = answer

    def action_answer(self, allowed: bool) -> None:
        """Tell the app what the user answered."""
        self.post_message(self.Answered(self, allowed))
