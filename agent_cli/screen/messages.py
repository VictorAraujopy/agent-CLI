"""The blocks of the conversation on the screen: the user's message, the model's thinking and reply, the notices."""

from markdown_it import MarkdownIt
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownStream

from agent_cli.inference.model import THINKING_CLOSES, THINKING_OPENS, thinking_and_answer
from agent_cli.screen.permission import PermissionPrompt
from agent_cli.screen.theme import MUTED_MARKUP
from agent_cli.tools.tool import ToolResult
from agent_cli.tools.toolbox import ToolUse


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


def thinking_and_answer_so_far(written: str) -> tuple[str, str]:
    """Split what the model wrote so far into thinking and answer, leaving out a tag still being written."""
    if THINKING_OPENS.startswith(written.lstrip()):
        return "", ""
    thinking, answer = thinking_and_answer(written)
    if THINKING_CLOSES not in written:
        thinking = without_a_tag_half_written(thinking, THINKING_CLOSES).rstrip()
    return thinking, answer


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


def style_of_preview_line(line: str) -> str:
    """Return the color of a line of a tool's preview: + comes into a file, - leaves it, the rest is muted."""
    if line.startswith("+ "):
        return "$success"
    if line.startswith("- "):
        return "$error"
    return MUTED_MARKUP


def result_lines(summary: str) -> str:
    """Return the summary of a tool's result as it is shown: its first line after the mark, the rest under it."""
    first_line, *other_lines = summary.splitlines() or [""]
    lines = [f"⎿ {first_line}"] + [f"  {line}" for line in other_lines]
    return "\n".join(lines)


class ToolView(Vertical):
    """A tool a reply asked for: its name and what it acts on, what it would change, and how the call ended."""

    def __init__(self, use: ToolUse) -> None:
        """Keep the call, as the toolbox read it."""
        super().__init__()
        self.use = use

    def compose(self) -> ComposeResult:
        """Yield the name of the tool with what it acts on and, for a tool that changes a file, the lines it changes."""
        yield Static(Content.assemble(("● ", "$primary"), (self.use.name, "bold"), "  ", self.use.summary))
        if self.use.preview:
            preview_lines = self.use.preview.splitlines()
            styled = [Content.styled(line, style_of_preview_line(line)) for line in preview_lines]
            yield Static(Content("\n").join(styled), classes="tool-preview")

    async def show_result(self, result: ToolResult) -> None:
        """Show under the tool how the call ended: the summary of its result, in the color of its ending."""
        ending = result.ending.name.lower()
        await self.mount(Static(result_lines(result.summary), classes=f"tool-result {ending}", markup=False))


class TurnView(Vertical):
    """One turn as the screen shows it: the user's message, and under it each reply and the tools it ran."""

    def __init__(self, user_text: str) -> None:
        """Keep the user's text and start with nothing written by the model."""
        super().__init__()
        self.user_text = user_text
        # Everything the model wrote in the reply being shown, thinking included, as it arrived.
        self.written = ""
        self.thinking = Thinking()
        self.reply = Reply()
        # The tool shown last. A piece that arrives after a tool belongs to the next reply, shown under it.
        self.last_tool: ToolView | None = None
        self.reply_is_over = False

    def compose(self) -> ComposeResult:
        """Yield the user's message and, under it, the model's thinking and its reply."""
        yield UserMessage(self.user_text)
        yield self.thinking
        yield self.reply

    async def show_search(self, text: str, tone: str) -> None:
        """Show, between the user's message and the first reply, how the search of earlier turns went."""
        await self.mount(Notice(text, tone), before=self.thinking)

    async def add_piece(self, piece: str) -> None:
        """Add one piece of what the model writes: to the thinking, or to the reply once the thinking is over."""
        if self.reply_is_over:
            await self.start_next_reply()
        answer_before = thinking_and_answer_so_far(self.written)[1]
        self.written += piece
        thinking, answer = thinking_and_answer_so_far(self.written)
        self.thinking.show(thinking)
        await self.reply.add_piece(answer[len(answer_before) :])

    async def start_next_reply(self) -> None:
        """Show the next reply of the turn, with its own thinking, under the tools the last one ran."""
        self.written = ""
        self.thinking = Thinking()
        self.reply = Reply()
        self.reply_is_over = False
        await self.mount(self.thinking, self.reply)

    async def add_tool(self, use: ToolUse) -> None:
        """Draw what is still waiting of the reply, and show under it the tool it asked for."""
        await self.reply.finish()
        self.reply_is_over = True
        self.last_tool = ToolView(use)
        await self.mount(self.last_tool)

    async def ask_permission(self, prompt: PermissionPrompt) -> None:
        """Show, under the tool shown last, the line that asks whether it may run."""
        if self.last_tool is not None:
            await self.last_tool.mount(prompt)

    async def show_tool_result(self, result: ToolResult) -> None:
        """Show how the call of the tool shown last ended."""
        if self.last_tool is not None:
            await self.last_tool.show_result(result)

    async def finish(self) -> None:
        """Draw what is still waiting; the turn gets no more pieces after this."""
        await self.reply.finish()
