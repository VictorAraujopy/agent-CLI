"""The blocks of the conversation on the screen: the user's message, the model's thinking and reply, the notices."""

from markdown_it import MarkdownIt
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownStream

from agent_cli.inference.model import THINKING_CLOSES, THINKING_OPENS


class UserMessage(Horizontal):
    """A message the user sent: a mark on the left and the text beside it."""

    def __init__(self, text: str) -> None:
        """Keep the text to show."""
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        """Yield the mark and the text, shown exactly as it was typed."""
        yield Static(">", classes="user-mark")
        # markup=False: a square bracket the user typed is text, not the start of a style tag.
        yield Static(self.text, classes="user-text", markup=False)


def parser_of_a_reply() -> MarkdownIt:
    """Return the Markdown parser of a reply: the one Textual uses, but with HTML tags kept as the text they are."""
    # The parser Textual uses reads what is written between < and > as an HTML tag, and Textual draws
    # nothing for a tag: a <div> the model explains, or a type such as vector<int>, would vanish from
    # the reply on the screen.
    return MarkdownIt("gfm-like", {"html": False})


class Reply(Markdown):
    """The model's reply, drawn as Markdown, which grows as its pieces arrive."""

    def __init__(self) -> None:
        """Start an empty reply, with nothing drawing it yet."""
        super().__init__(parser_factory=parser_of_a_reply)
        # Draws the pieces in the background, joining the ones that arrive faster than the screen can draw.
        # It starts with the first piece, because it needs the app running.
        self.stream: MarkdownStream | None = None

    async def add_piece(self, piece: str) -> None:
        """Add one piece of text to the end of the reply."""
        if self.stream is None:
            self.stream = Markdown.get_stream(self)
        await self.stream.write(piece)

    async def finish(self) -> None:
        """Draw the pieces still waiting; the reply gets no more pieces after this."""
        if self.stream is not None:
            await self.stream.stop()


def without_a_tag_half_written(text: str, tag: str) -> str:
    """Return the text without its end, when that end is the start of the tag, which is still being written."""
    for letters_written in range(len(tag) - 1, 0, -1):
        if text.endswith(tag[:letters_written]):
            return text[:-letters_written]
    return text


def thinking_and_answer(written: str) -> tuple[str, str]:
    """Split what the model wrote so far into what it thought and what it answered."""
    start = written.lstrip()
    if not start.startswith(THINKING_OPENS):
        # While the first letters can still become the opening tag, they are neither thinking nor answer yet.
        return ("", "") if THINKING_OPENS.startswith(start) else ("", written)
    thinking, closing, answer = start.removeprefix(THINKING_OPENS).partition(THINKING_CLOSES)
    if not closing:
        thinking = without_a_tag_half_written(thinking, THINKING_CLOSES)
    return thinking.strip(), answer.lstrip()


class Thinking(Static):
    """What the model thought before it answered, shown above the reply, hidden while there is none."""

    def __init__(self) -> None:
        """Start with no thinking to show."""
        super().__init__(markup=False)
        self.shown = ""

    def show(self, thinking: str) -> None:
        """Show the thinking written so far, if it changed."""
        if thinking != self.shown:
            self.shown = thinking
            self.update(thinking)
            self.display = True


class Notice(Static):
    """One line the screen itself says about a message or a reply, in the tone its class gives it."""

    def __init__(self, text: str, tone: str) -> None:
        """Keep the text, shown as it is, and the tone, which the stylesheet turns into a color."""
        super().__init__(text, classes=tone, markup=False)


class TurnView(Vertical):
    """One turn as the screen shows it: the user's message and, under it, the model's thinking and its reply."""

    def __init__(self, user_text: str) -> None:
        """Keep the user's text and start with nothing written by the model."""
        super().__init__()
        self.user_text = user_text
        # Everything the model wrote in this turn, thinking included, as it arrived.
        self.written = ""
        self.thinking = Thinking()
        self.reply = Reply()

    def compose(self) -> ComposeResult:
        """Yield the user's message and, under it, the model's thinking and its reply."""
        yield UserMessage(self.user_text)
        yield self.thinking
        yield self.reply

    async def add_piece(self, piece: str) -> None:
        """Add one piece of what the model writes: to the thinking, or to the reply once the thinking is over."""
        answer_before = thinking_and_answer(self.written)[1]
        self.written += piece
        thinking, answer = thinking_and_answer(self.written)
        self.thinking.show(thinking)
        await self.reply.add_piece(answer[len(answer_before) :])

    async def finish(self) -> None:
        """Draw what is still waiting; the turn gets no more pieces after this."""
        await self.reply.finish()
