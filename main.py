"""Starts agent-CLI: reads the settings, prepares how each configured model is opened and opens the screen."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from agent_cli.agent.conversation import Conversation
from agent_cli.agent.prompt import read_system_prompt
from agent_cli.config import Settings, check_models_configured, load_settings
from agent_cli.history.queries import History
from agent_cli.inference.api import open_api_model, open_client
from agent_cli.inference.catalog import Catalog
from agent_cli.inference.generation import open_local_model
from agent_cli.inference.model import Model
from agent_cli.screen.app import AgentCliApp
from agent_cli.tools.toolbox import Toolbox


def models_configured(settings: Settings) -> dict[str, Callable[[], Model]]:
    """Return, for the name of each model configured, the function that opens it: the local one first."""
    how_to_open: dict[str, Callable[[], Model]] = {}
    if settings.model_path is not None:
        local_name = settings.model_path.stem
        how_to_open[local_name] = partial(open_local_model, settings.model_path, settings.rpc_servers)
    if settings.api_models:
        client = open_client(settings.api_key)
        for name in settings.api_models:
            how_to_open[name] = partial(open_api_model, client, name)
    return how_to_open


def arguments_given() -> argparse.Namespace:
    """Read what agentcli was called with: -r opens the list of the earlier conversations of the folder."""
    parser = argparse.ArgumentParser(prog="agentcli", description="Terminal agent for GGUF or API models.")
    parser.add_argument(
        "-r", "--resume", action="store_true", help="choose an earlier conversation of this folder to continue"
    )
    return parser.parse_args()


def history_file() -> Path:
    """Return the file that keeps the conversations of every folder, in the user's home folder."""
    return Path.home() / ".agent-cli" / "history.db"


def main() -> None:
    """Read the settings, run the session on the screen and end with the exit code the screen closed with."""
    arguments = arguments_given()
    settings = load_settings()
    check_models_configured(settings)
    how_to_open = models_configured(settings)
    system_prompt = read_system_prompt(settings.system_prompt_path)
    # The tools act from the folder the user called agentcli in, as a shell command would, and the
    # conversations are kept by that folder.
    working_folder = Path.cwd()
    toolbox = Toolbox(working_folder)
    history = History(history_file(), working_folder)
    conversation = Conversation(system_prompt, toolbox, history)
    catalog = Catalog(how_to_open)
    app = AgentCliApp(conversation, catalog, resume=arguments.resume)
    try:
        app.run()
    finally:
        # A local model lives in a llama-server of its own, which would keep its memory after the screen closes.
        catalog.close()
    raise SystemExit(app.return_code)


if __name__ == "__main__":
    main()
