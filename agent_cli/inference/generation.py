"""Generation with a local model: LocalModel, the Model served by the llama-server this session started."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from agent_cli.inference.chat_completions import ChatCompletionsModel, connection_failure, error_in_the_body
from agent_cli.inference.loader import LocalServer, load_model, stop_server
from agent_cli.inference.model import ContextFullError


def open_local_client(server: LocalServer) -> httpx.Client:
    """Return the HTTP client that reaches the session's llama-server, sending the key it asks for."""
    # While it reads a conversation the server sends its progress at every batch, so two minutes with
    # nothing arriving means it hung; a server that ended closes the connection instead.
    return httpx.Client(
        base_url=server.address,
        headers={"Authorization": f"Bearer {server.key}"},
        timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
        # A proxy set in the environment would get the key and the conversation, which never leave this machine.
        trust_env=False,
    )


def failure_reported(error: dict[str, Any]) -> Exception:
    """Return the exception for an error llama-server reported: ContextFullError when the conversation does not fit."""
    message = error.get("message", "no reason given")
    if error.get("type") == "exceed_context_size_error":
        return ContextFullError(message)
    return RuntimeError(f"llama-server reported an error: {message}")


def context_tokens_of(client: httpx.Client) -> int:
    """Ask the server the size of the context it created: fitted to the free memory, never above the trained one."""
    try:
        answer = client.get("/props")
    except httpx.TransportError as error:
        raise connection_failure("llama-server", error) from error
    if answer.status_code != 200:
        raise failure_reported(error_in_the_body(answer))
    return answer.json()["default_generation_settings"]["n_ctx"]


def open_local_model(model_path: Path, rpc_servers: Sequence[str] = ()) -> "LocalModel":
    """Start llama-server with the .gguf file, alone or split with the RPC servers, and return the model it serves."""
    server = load_model(model_path, rpc_servers)
    client = open_local_client(server)
    try:
        context_tokens = context_tokens_of(client)
    except BaseException:
        client.close()
        stop_server(server)
        raise
    return LocalModel(server, client, context_tokens)


class LocalModel(ChatCompletionsModel):
    """A model running on this machine, and on the RPC servers when split, inside the session's llama-server."""

    server_name = "llama-server"
    thinking_field = "reasoning_content"

    def __init__(self, server: LocalServer, client: httpx.Client, context_tokens: int) -> None:
        """Keep the server that holds the model and the client that reaches it, with the thinking on."""
        super().__init__(client, context_tokens)
        self.server = server
        self.thinking_is_on = True

    def fields_of_this_server(self) -> dict[str, Any]:
        """Return what llama-server needs besides the conversation: the thinking switch, the count and the progress."""
        return {
            # The chat template of the .gguf reads this switch; a template that has none ignores it.
            "chat_template_kwargs": {"enable_thinking": self.thinking_is_on},
            # The count of tokens in every chunk, so that a reply stopped halfway is counted too.
            "timings_per_token": True,
            # A chunk at every batch of a long conversation the server reads, so that a stop is seen meanwhile.
            "return_progress": True,
        }

    def failure_from(self, error: dict[str, Any]) -> Exception:
        """Return the exception for an error llama-server reported: ContextFullError when the conversation does not fit."""
        return failure_reported(error)

    def tokens_counted_in(self, chunk: dict[str, Any]) -> int | None:
        """Return the tokens in the context: the whole conversation while the server reads it, then the reply as well."""
        progress = chunk.get("prompt_progress")
        if progress is not None:
            return progress["total"]
        timings = chunk.get("timings")
        if timings is None:
            return None
        return timings["cache_n"] + timings["prompt_n"] + timings["predicted_n"]

    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices for the thinking of a local model: on or off, the switch a chat template has."""
        return ("on", "off")

    def thinking(self) -> str:
        """Return whether the model may think before the replies that follow: on or off."""
        return "on" if self.thinking_is_on else "off"

    def set_thinking(self, option: str) -> None:
        """Choose whether the model thinks before the replies that follow: on or off."""
        self.thinking_is_on = option == "on"

    def close(self) -> None:
        """End the server, which frees the memory the model takes, on this machine and on the RPC servers."""
        self.client.close()
        stop_server(self.server)
