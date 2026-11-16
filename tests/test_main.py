"""Tests for the start of the program: the models it configures, the session it builds, and how the program ends."""

from pathlib import Path

import pytest

import main
from agent_cli.config import Settings
from agent_cli.inference.generation import LocalModel

SMALL_MODEL_TRAINED_CONTEXT = 2048


class CatalogThatRemembersBeingClosed:
    """Stands in for the catalog of a session: closing it only remembers that it was closed."""

    was_closed = False

    def close(self) -> None:
        """Close nothing: only remember that the catalog was closed."""
        self.was_closed = True


class ScreenThatClosedWithAnError:
    """Stands in for the screen of a session that failed: running it opens nothing, and its exit code is 1."""

    return_code = 1
    was_run = False

    def __init__(self) -> None:
        """Start with a catalog that remembers whether the program closed it."""
        self.catalog = CatalogThatRemembersBeingClosed()

    def run(self) -> None:
        """Open nothing: only remember that the session was run."""
        self.was_run = True


class ScreenThatBreaks(ScreenThatClosedWithAnError):
    """Stands in for a screen that fails with an error while it runs."""

    def run(self) -> None:
        """Fail as a screen that broke would."""
        raise RuntimeError("the screen broke")


def settings_with(model_path: Path | None) -> Settings:
    """Return settings that configure nothing but the path of the local model."""
    return Settings(model_path=model_path, rpc_servers=(), api_key=None, api_models=(), system_prompt_path=None)


def settings_for(model_path: Path | None, api_key: str | None, api_models: tuple[str, ...]) -> Settings:
    """Return settings with the given local model, key and API models, and nothing else."""
    return Settings(
        model_path=model_path,
        rpc_servers=("10.0.0.2:50052",),
        api_key=api_key,
        api_models=api_models,
        system_prompt_path=None,
    )


def test_session_with_no_model_configured_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no local model and no API model the program ends before opening the screen, naming the variables."""
    monkeypatch.setattr(main, "load_settings", lambda: settings_with(None))
    with pytest.raises(SystemExit, match="AGENT_CLI_MODEL_PATH or AGENT_CLI_API_MODELS"):
        main.main()


def test_api_models_without_a_key_are_refused() -> None:
    """API models configured with no key end the program with the variable to set, instead of vanishing."""
    with pytest.raises(SystemExit, match="AGENT_CLI_API_KEY"):
        main.models_configured(settings_for(None, None, ("author/model",)))


def test_local_model_comes_first_and_the_api_models_after_it_in_their_order() -> None:
    """The session starts on the first name: the local model when there is one, then the API models as written."""
    settings = settings_for(Path("folder/local-model.gguf"), "the-key", ("author/one", "author/two"))
    assert list(main.models_configured(settings)) == ["local-model", "author/one", "author/two"]
    assert list(main.models_configured(settings_for(None, "the-key", ("author/one",)))) == ["author/one"]


def test_each_model_is_opened_with_what_the_settings_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local model opens with its file and RPC servers; each API model opens by name, all with one client."""
    opened: list[tuple[object, ...]] = []
    monkeypatch.setattr(main, "open_local_model", lambda *parts: opened.append(("local", *parts)))
    monkeypatch.setattr(main, "open_api_model", lambda client, name: opened.append(("api", client, name)))
    monkeypatch.setattr(main, "open_client", lambda api_key: f"client of {api_key}")
    how_to_open = main.models_configured(settings_for(Path("any.gguf"), "the-key", ("author/one", "author/two")))
    for open_model in how_to_open.values():
        open_model()
    assert opened == [
        ("local", Path("any.gguf"), ("10.0.0.2:50052",)),
        ("api", "client of the-key", "author/one"),
        ("api", "client of the-key", "author/two"),
    ]


def test_program_ends_with_the_exit_code_of_the_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session that closed with an error makes the program end with that error code."""
    monkeypatch.setattr(main, "load_settings", lambda: settings_with(Path("any.gguf")))
    monkeypatch.setattr(main, "build_app", lambda *parts: ScreenThatClosedWithAnError())
    with pytest.raises(SystemExit) as ending:
        main.main()
    assert ending.value.code == 1


def test_program_runs_the_session_built_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session that runs is built from the models configured and the system prompt file configured."""
    settings = Settings(
        model_path=Path("any.gguf"), rpc_servers=(), api_key=None, api_models=(), system_prompt_path=Path("prompt.md")
    )
    built_from: list[tuple[object, ...]] = []
    screen = ScreenThatClosedWithAnError()

    def build(how_to_open: dict[str, object], system_prompt_path: Path) -> ScreenThatClosedWithAnError:
        """Stand in for the building of the session: record what it is built from and hand over the screen."""
        built_from.append((list(how_to_open), system_prompt_path))
        return screen

    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "build_app", build)
    with pytest.raises(SystemExit):
        main.main()
    assert built_from == [(["any"], Path("prompt.md"))]
    assert screen.was_run


@pytest.mark.parametrize("screen", [ScreenThatClosedWithAnError(), ScreenThatBreaks()])
def test_model_in_use_is_closed_when_the_screen_closes(
    screen: ScreenThatClosedWithAnError, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the screen closes or breaks, the catalog is closed, which ends the server of a local model."""
    monkeypatch.setattr(main, "load_settings", lambda: settings_with(Path("any.gguf")))
    monkeypatch.setattr(main, "build_app", lambda *parts: screen)
    with pytest.raises((SystemExit, RuntimeError)):
        main.main()
    assert screen.catalog.was_closed


def test_session_is_built_with_the_system_prompt_and_the_models(small_model_file: Path, tmp_path: Path) -> None:
    """The screen gets a conversation with the configured system prompt and a catalog that opens the models."""
    system_prompt_file = tmp_path / "system_prompt.md"
    system_prompt_file.write_text("Answer in one line.", encoding="utf-8")
    how_to_open = main.models_configured(settings_with(small_model_file))
    app = main.build_app(how_to_open, system_prompt_file)
    assert app.conversation.messages() == [{"role": "system", "content": "Answer in one line."}]
    assert app.catalog.names() == [small_model_file.stem]
    model = app.catalog.open(small_model_file.stem)
    try:
        assert isinstance(model, LocalModel)
        assert model.context_limit() == SMALL_MODEL_TRAINED_CONTEXT
    finally:
        app.catalog.close()
    assert model.server.process.poll() is not None
