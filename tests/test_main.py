"""Tests for the start of the program: the models it configures, the session it builds, and how the program ends."""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

import main
from agent_cli.agent.conversation import Conversation
from agent_cli.config import Settings
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.generation import LocalModel
from agent_cli.inference.model import Model

SMALL_MODEL_TRAINED_CONTEXT = 2048


class CatalogThatRemembersBeingClosed:
    """Stands in for the catalog of a session: closing it only remembers that it was closed."""

    def __init__(self, how_to_open: dict[str, Callable[[], Model]]) -> None:
        """Take the models as the real catalog does, keep none of them, and start open."""
        self.was_closed = False

    def close(self) -> None:
        """Close nothing: only remember that the catalog was closed."""
        self.was_closed = True


class ScreenThatClosedWithAnError:
    """Stands in for the screen of a session that failed: running it opens nothing, and its exit code is 1."""

    return_code = 1

    def __init__(self, conversation: Conversation, catalog: Catalog, resume: bool = False) -> None:
        """Keep the conversation, the catalog and the choice to resume the program built the session with."""
        self.conversation = conversation
        self.catalog = catalog
        self.resume = resume
        self.was_run = False

    def run(self) -> None:
        """Open nothing: only remember that the session was run."""
        self.was_run = True


class ScreenThatBreaks(ScreenThatClosedWithAnError):
    """Stands in for a screen that fails with an error while it runs."""

    def run(self) -> None:
        """Fail as a screen that broke would."""
        raise RuntimeError("the screen broke")


class ScreenThatOpensTheFirstModel(ScreenThatClosedWithAnError):
    """Stands in for a session that opens its first model, as the screen does when it starts, and closes well."""

    return_code = 0

    def run(self) -> None:
        """Open the first model of the catalog and keep it, so the test can look at it after the session."""
        first_name = self.catalog.names()[0]
        self.model_opened = self.catalog.open(first_name)


def settings_with(model_path: Path | None) -> Settings:
    """Return settings that configure nothing but the path of the local model."""
    return Settings(
        model_path=model_path,
        rpc_servers=(),
        api_key=None,
        api_models=(),
        system_prompt_path=None,
        embedding_model=None,
    )


def settings_for(model_path: Path | None, api_key: str | None, api_models: tuple[str, ...]) -> Settings:
    """Return settings with the given local model, key and API models, and nothing else."""
    return Settings(
        model_path=model_path,
        rpc_servers=("10.0.0.2:50052",),
        api_key=api_key,
        api_models=api_models,
        system_prompt_path=None,
        embedding_model=None,
    )


def screens_built_by_main(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    screen_class: type[ScreenThatClosedWithAnError],
    arguments: tuple[str, ...] = (),
) -> list[ScreenThatClosedWithAnError]:
    """Make the program read the settings and the arguments, and build its screen from the class; return the screens."""
    screens_built: list[ScreenThatClosedWithAnError] = []

    def build_screen(conversation: Conversation, catalog: Catalog, resume: bool) -> ScreenThatClosedWithAnError:
        """Build the stand-in screen and keep it, so the test can look at it after the program ends."""
        screen = screen_class(conversation, catalog, resume)
        screens_built.append(screen)
        return screen

    monkeypatch.setattr(sys, "argv", ["agentcli", *arguments])
    monkeypatch.setattr(main, "history_file", lambda: Path(tempfile.mkdtemp()) / "history.db")
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "AgentCliApp", build_screen)
    return screens_built


def test_session_with_no_model_configured_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no local model and no API model the program ends before building the screen, naming the variables."""
    screens = screens_built_by_main(monkeypatch, settings_with(None), ScreenThatClosedWithAnError)
    with pytest.raises(SystemExit, match="AGENT_CLI_MODEL_PATH or AGENT_CLI_API_MODELS"):
        main.main()
    assert screens == []


def test_local_model_comes_first_and_the_api_models_after_it_in_their_order() -> None:
    """The session starts on the first name: the local model when there is one, then the API models as written."""
    settings = settings_for(Path("folder/local-model.gguf"), "the-key", ("author/one", "author/two"))
    assert list(main.models_configured(settings)) == ["local-model", "author/one", "author/two"]
    assert list(main.models_configured(settings_for(None, "the-key", ("author/one",)))) == ["author/one"]


def test_each_model_is_opened_with_what_the_settings_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local model opens with its file and RPC servers; each API model opens by name, all with one client."""
    opened: list[tuple[object, ...]] = []
    clients_opened: list[str] = []

    def open_one_more_client(api_key: str) -> str:
        """Stand in for the client of OpenRouter, numbered by how many were opened."""
        clients_opened.append(api_key)
        return f"client {len(clients_opened)} of {api_key}"

    monkeypatch.setattr(main, "open_local_model", lambda *parts: opened.append(("local", *parts)))
    monkeypatch.setattr(main, "open_api_model", lambda client, name: opened.append(("api", client, name)))
    monkeypatch.setattr(main, "open_client", open_one_more_client)
    how_to_open = main.models_configured(settings_for(Path("any.gguf"), "the-key", ("author/one", "author/two")))
    for open_model in how_to_open.values():
        open_model()
    assert opened == [
        ("local", Path("any.gguf"), ("10.0.0.2:50052",)),
        ("api", "client 1 of the-key", "author/one"),
        ("api", "client 1 of the-key", "author/two"),
    ]


def test_program_ends_with_the_exit_code_of_the_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session that closed with an error makes the program end with that error code."""
    screens_built_by_main(monkeypatch, settings_with(Path("any.gguf")), ScreenThatClosedWithAnError)
    with pytest.raises(SystemExit) as ending:
        main.main()
    assert ending.value.code == 1


def test_program_runs_the_session_built_from_the_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The session runs with the models and the system prompt configured, and tools acting from the current folder."""
    settings = Settings(
        model_path=Path("any.gguf"),
        rpc_servers=(),
        api_key=None,
        api_models=(),
        system_prompt_path=Path("prompt.md"),
        embedding_model=None,
    )
    monkeypatch.setattr(main, "read_system_prompt", lambda prompt_file: f"the prompt in {prompt_file}")
    screens = screens_built_by_main(monkeypatch, settings, ScreenThatClosedWithAnError)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main.main()
    [screen] = screens
    assert screen.conversation.messages() == [{"role": "system", "content": "the prompt in prompt.md"}]
    assert screen.conversation.toolbox.tools["run_command"].working_folder == tmp_path.resolve()
    assert screen.catalog.names() == ["any"]
    assert screen.was_run


@pytest.mark.parametrize("screen_class", [ScreenThatClosedWithAnError, ScreenThatBreaks])
def test_model_in_use_is_closed_when_the_screen_closes(
    screen_class: type[ScreenThatClosedWithAnError], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the screen closes or breaks, the catalog is closed, which ends the server of a local model."""
    monkeypatch.setattr(main, "Catalog", CatalogThatRemembersBeingClosed)
    screens = screens_built_by_main(monkeypatch, settings_with(Path("any.gguf")), screen_class)
    with pytest.raises((SystemExit, RuntimeError)):
        main.main()
    [screen] = screens
    assert screen.catalog.was_closed


def test_session_opens_its_models_and_closes_them_when_it_ends(
    small_model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session gets the configured system prompt and a catalog that opens the model, whose server ends with it."""
    system_prompt_file = tmp_path / "system_prompt.md"
    system_prompt_file.write_text("Answer in one line.", encoding="utf-8")
    settings = Settings(
        model_path=small_model_file,
        rpc_servers=(),
        api_key=None,
        api_models=(),
        system_prompt_path=system_prompt_file,
        embedding_model=None,
    )
    screens = screens_built_by_main(monkeypatch, settings, ScreenThatOpensTheFirstModel)
    with pytest.raises(SystemExit) as ending:
        main.main()
    [screen] = screens
    assert ending.value.code == 0
    assert screen.conversation.messages() == [{"role": "system", "content": "Answer in one line."}]
    assert isinstance(screen.model_opened, LocalModel)
    assert screen.model_opened.context_limit() == SMALL_MODEL_TRAINED_CONTEXT
    assert screen.model_opened.server.process.poll() is not None


@pytest.mark.parametrize(("arguments", "resume"), [((), False), (("-r",), True), (("--resume",), True)])
def test_r_opens_the_session_on_the_list_of_earlier_conversations(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...], resume: bool
) -> None:
    """agentcli -r, or --resume, asks the screen to offer the earlier conversations first."""
    settings = settings_with(Path("any.gguf"))
    screens = screens_built_by_main(monkeypatch, settings, ScreenThatClosedWithAnError, arguments)
    with pytest.raises(SystemExit):
        main.main()
    assert screens[0].resume is resume


def test_unknown_argument_ends_the_program_before_anything_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """An argument agentcli does not know ends it with the usage, before the settings are read."""
    screens = screens_built_by_main(monkeypatch, settings_with(Path("any.gguf")), ScreenThatClosedWithAnError, ("-x",))
    monkeypatch.setattr(main, "load_settings", lambda: pytest.fail("the settings were read"))
    with pytest.raises(SystemExit) as ending:
        main.main()
    assert ending.value.code == 2
    assert screens == []


def test_history_is_kept_in_the_home_folder_by_the_folder_agentcli_runs_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One file in the user's home folder keeps every conversation, each under the folder it happened in."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert main.history_file() == tmp_path / "home" / ".agent-cli" / "history.db"
    screens = screens_built_by_main(monkeypatch, settings_with(Path("any.gguf")), ScreenThatClosedWithAnError)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main.main()
    assert screens[0].conversation.history.working_folder == str(tmp_path.resolve())


def test_session_with_a_key_but_no_embedding_model_searches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key alone does not turn the search on, so the turns of a local model never leave the machine."""
    settings = settings_for(Path("any.gguf"), api_key="the-key", api_models=())
    screens = screens_built_by_main(monkeypatch, settings, ScreenThatClosedWithAnError)
    with pytest.raises(SystemExit):
        main.main()
    assert screens[0].conversation.recall is None


def test_session_searches_earlier_turns_with_the_embedding_model_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search asks the configured embedding model, with the key, and keeps its vectors in the session's history."""
    settings = Settings(
        model_path=Path("any.gguf"),
        rpc_servers=(),
        api_key="the-key",
        api_models=(),
        system_prompt_path=None,
        embedding_model="author/embedding",
    )
    screens = screens_built_by_main(monkeypatch, settings, ScreenThatClosedWithAnError)
    with pytest.raises(SystemExit):
        main.main()
    [screen] = screens
    recall = screen.conversation.recall
    assert recall.embedding_model.model_name == "author/embedding"
    assert recall.embedding_model.client.headers["Authorization"] == "Bearer the-key"
    assert recall.history is screen.conversation.history
