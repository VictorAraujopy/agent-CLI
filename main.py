"""Starts agent-CLI: reads the settings, sizes the model's context to the free memory and opens the screen."""

from pathlib import Path

from agent_cli.agent.conversation import Conversation
from agent_cli.agent.prompt import read_system_prompt
from agent_cli.config import load_settings
from agent_cli.inference.generation import LocalModel
from agent_cli.inference.loader import DeviceMemory, MemoryUse, load_model, measure_memory_use
from agent_cli.inference.model import Model
from agent_cli.screen.app import AgentCliApp


def tokens_that_fit_on(device: DeviceMemory) -> int:
    """Return how many tokens of context fit in what the device has free once the model's weights are on it."""
    memory_left_free = 512 * 1024**2
    memory_for_the_context = (
        device.free_before_the_model - memory_left_free - device.model_weights - device.any_context
    )
    return int(memory_for_the_context / device.each_context_token)


def context_that_fits(memory_use: MemoryUse) -> int:
    """Return the largest context every device holds, in tokens, never above the one the model was trained with."""
    context = memory_use.trained_context
    for device in memory_use.devices:
        context = min(context, tokens_that_fit_on(device))
    if context < memory_use.trained_context:
        context = context // 256 * 256
    smallest_acceptable = min(memory_use.trained_context, 4096)
    if context < smallest_acceptable:
        raise RuntimeError(
            "the model does not fit in the free memory: "
            f"there is room for a context of {max(context, 0)} tokens, and it needs at least {smallest_acceptable}"
        )
    return context


def open_local_model(model_path: Path, rpc_servers: tuple[str, ...]) -> Model:
    """Size the context to the free memory, on this machine and on the RPC servers, and load the .gguf with it."""
    memory_use = measure_memory_use(model_path, rpc_servers)
    context_tokens = context_that_fits(memory_use)
    llama = load_model(model_path, rpc_servers, context_tokens)
    return LocalModel(llama)


def build_app(model_path: Path, rpc_servers: tuple[str, ...], system_prompt_path: Path | None) -> AgentCliApp:
    """Prepare the screen of a session with the local model: its conversation, its name and how to open it."""
    system_prompt = read_system_prompt(system_prompt_path)
    return AgentCliApp(
        conversation=Conversation(system_prompt),
        model_name=model_path.stem,
        # A function, not the model: the screen calls it in the background, so it opens before the model is loaded.
        open_model=lambda: open_local_model(model_path, rpc_servers),
    )


def main() -> None:
    """Read the settings, run the session on the screen and end with the exit code the screen closed with."""
    settings = load_settings()
    if settings.model_path is None:
        raise SystemExit("agent-CLI: no local model configured. Set AGENT_CLI_MODEL_PATH in the .env file.")
    app = build_app(settings.model_path, settings.rpc_servers, settings.system_prompt_path)
    app.run()
    raise SystemExit(app.return_code)


if __name__ == "__main__":
    main()
