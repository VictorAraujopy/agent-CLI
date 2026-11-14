"""Tests for the start of the program: the context it sizes, the session it builds, and how the program ends."""

from pathlib import Path

import pytest

import main
from agent_cli.config import Settings
from agent_cli.inference.generation import LocalModel
from agent_cli.inference.loader import DeviceMemory, MemoryUse

SMALL_MODEL_TRAINED_CONTEXT = 2048
GIB = 1024**3


class ScreenThatClosedWithAnError:
    """Stands in for the screen of a session that failed: running it opens nothing, and its exit code is 1."""

    return_code = 1
    was_run = False

    def run(self) -> None:
        """Open nothing: only remember that the session was run."""
        self.was_run = True


def settings_with(model_path: Path | None) -> Settings:
    """Return settings that configure nothing but the path of the local model."""
    return Settings(model_path=model_path, rpc_servers=(), api_key=None, api_models=(), system_prompt_path=None)


def device_with(free_gib: float, weights_gib: float = 0, any_context_gib: float = 0) -> DeviceMemory:
    """Return a device with the given memory, in GiB, on which each token of context takes 1 MiB."""
    return DeviceMemory(
        name="GPU0 (test)",
        free_before_the_model=round(free_gib * GIB),
        model_weights=round(weights_gib * GIB),
        any_context=round(any_context_gib * GIB),
        each_context_token=float(1024**2),
    )


def test_tokens_that_fit_come_from_what_the_weights_any_context_and_the_reserve_leave() -> None:
    """Of 10 GiB free, 2 go to the weights, 1 to any context and half stays free: 6.5 GiB hold 6656 tokens of 1 MiB."""
    assert main.tokens_that_fit_on(device_with(free_gib=10, weights_gib=2, any_context_gib=1)) == 6656


def test_model_that_fits_whole_keeps_the_context_it_was_trained_with() -> None:
    """When the memory holds more tokens than the model was trained with, the trained context is the one used."""
    memory_use = MemoryUse(trained_context=5000, devices=(device_with(free_gib=100),))
    assert main.context_that_fits(memory_use) == 5000


def test_context_is_cut_to_what_the_memory_holds_in_steps_of_256_tokens() -> None:
    """Room for 6244 tokens gives a context of 6144: the largest multiple of 256 that still fits."""
    room_for_6244_tokens = device_with(free_gib=0.5 + 6244 / 1024)
    assert main.context_that_fits(MemoryUse(trained_context=262144, devices=(room_for_6244_tokens,))) == 6144


def test_device_with_the_least_room_decides_the_context() -> None:
    """A model split across two devices gets the context the tighter one holds, here 5000 tokens cut to 4864."""
    roomy, tight = device_with(free_gib=0.5 + 20000 / 1024), device_with(free_gib=0.5 + 5000 / 1024)
    assert main.context_that_fits(MemoryUse(trained_context=262144, devices=(roomy, tight))) == 4864


def test_model_with_no_device_of_its_own_keeps_the_trained_context() -> None:
    """With no device holding the context, nothing limits it: the model gets the context it was trained with."""
    assert main.context_that_fits(MemoryUse(trained_context=32768, devices=())) == 32768


def test_model_that_leaves_too_little_room_for_a_context_is_refused() -> None:
    """Room for 1000 tokens is less than the smallest context worth loading: the error says the model does not fit."""
    room_for_1000_tokens = device_with(free_gib=0.5 + 1000 / 1024)
    with pytest.raises(RuntimeError, match="does not fit in the free memory.*768 tokens.*at least 4096"):
        main.context_that_fits(MemoryUse(trained_context=32768, devices=(room_for_1000_tokens,)))


def test_weights_larger_than_the_free_memory_are_refused() -> None:
    """A model whose weights alone pass the free memory has room for no context at all."""
    with pytest.raises(RuntimeError, match="a context of 0 tokens"):
        main.context_that_fits(MemoryUse(trained_context=32768, devices=(device_with(free_gib=4, weights_gib=8),)))


def test_model_trained_with_a_small_context_is_not_refused_for_it() -> None:
    """A model trained with fewer tokens than the smallest context worth loading is loaded with all of its own."""
    assert main.context_that_fits(MemoryUse(trained_context=2048, devices=(device_with(free_gib=100),))) == 2048


def test_local_model_is_loaded_with_the_context_that_fits(
    small_model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the local model measures it, sizes the context from the measurement and loads it with that context."""
    measurements_received = []

    def half_the_trained_context(memory_use: MemoryUse) -> int:
        """Stand in for the calculation: record what was measured and answer with a context of 1024 tokens."""
        measurements_received.append(memory_use)
        return 1024

    monkeypatch.setattr(main, "context_that_fits", half_the_trained_context)
    model = main.open_local_model(small_model_file, ())
    assert isinstance(model, LocalModel)
    assert model.context_limit() == 1024
    assert measurements_received[0].trained_context == SMALL_MODEL_TRAINED_CONTEXT


def test_local_model_is_measured_and_loaded_split_with_the_same_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RPC servers reach both the measurement and the load, so the context is sized for the split that is used."""
    asked: list[tuple[object, ...]] = []

    def measure(model_path: Path, rpc_servers: tuple[str, ...]) -> MemoryUse:
        """Stand in for the measurement: record what it was asked for and answer with a model that fits whole."""
        asked.append(("measure", model_path, rpc_servers))
        return MemoryUse(trained_context=8192, devices=())

    def load(model_path: Path, rpc_servers: tuple[str, ...], context_tokens: int) -> str:
        """Stand in for the load: record what it was asked for."""
        asked.append(("load", model_path, rpc_servers, context_tokens))
        return "the loaded model"

    monkeypatch.setattr(main, "measure_memory_use", measure)
    monkeypatch.setattr(main, "load_model", load)
    monkeypatch.setattr(main, "LocalModel", lambda llama: llama)
    assert main.open_local_model(Path("any.gguf"), ("10.0.0.2:50052",)) == "the loaded model"
    assert asked == [
        ("measure", Path("any.gguf"), ("10.0.0.2:50052",)),
        ("load", Path("any.gguf"), ("10.0.0.2:50052",), 8192),
    ]


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


def test_session_is_built_with_the_system_prompt_and_the_models(small_model_file: Path, tmp_path: Path) -> None:
    """The screen gets a conversation with the configured system prompt and a catalog that opens the models."""
    system_prompt_file = tmp_path / "system_prompt.md"
    system_prompt_file.write_text("Answer in one line.", encoding="utf-8")
    how_to_open = main.models_configured(settings_with(small_model_file))
    app = main.build_app(how_to_open, system_prompt_file)
    assert app.conversation.messages() == [{"role": "system", "content": "Answer in one line."}]
    assert app.catalog.names() == [small_model_file.stem]
    model = app.catalog.open(small_model_file.stem)
    assert isinstance(model, LocalModel)
    assert model.context_limit() == SMALL_MODEL_TRAINED_CONTEXT
