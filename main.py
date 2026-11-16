"""Starts agent-CLI: reads the settings, prepares how each configured model is opened and opens the screen."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from agent_cli.agent.conversation import Conversation
from agent_cli.agent.prompt import read_system_prompt
from agent_cli.config import Settings, load_settings
from agent_cli.inference.api import open_api_model, open_client
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.generation import open_local_model
from agent_cli.inference.model import Model
from agent_cli.screen.app import AgentCliApp


def models_configured(settings: Settings) -> dict[str, Callable[[], Model]]:
    """Return, for the name of each model configured, the function that opens it: the local one first."""
    how_to_open: dict[str, Callable[[], Model]] = {}
    if settings.model_path is not None:
        how_to_open[settings.model_path.stem] = partial(open_local_model, settings.model_path, settings.rpc_servers)
    if settings.api_models and settings.api_key is None:
        raise SystemExit("agent-CLI: API models are configured without a key. Set AGENT_CLI_API_KEY in the .env file.")
    if settings.api_models:
        client = open_client(settings.api_key)
        for name in settings.api_models:
            how_to_open[name] = partial(open_api_model, client, name)
    if not how_to_open:
        raise SystemExit("agent-CLI: no model configured. Set AGENT_CLI_MODEL_PATH or AGENT_CLI_API_MODELS in .env.")
    return how_to_open


def build_app(how_to_open: dict[str, Callable[[], Model]], system_prompt_path: Path | None) -> AgentCliApp:
    """Prepare the screen of a session: its conversation and the catalog of the models it can use."""
    system_prompt = read_system_prompt(system_prompt_path)
    return AgentCliApp(Conversation(system_prompt), Catalog(how_to_open))


def main() -> None:
    """Read the settings, run the session on the screen and end with the exit code the screen closed with."""
    settings = load_settings()
    app = build_app(models_configured(settings), settings.system_prompt_path)
    try:
        app.run()
    finally:
        # A local model lives in a llama-server of its own, which would keep its memory after the screen closes.
        app.catalog.close()
    raise SystemExit(app.return_code)


if __name__ == "__main__":
    main()
