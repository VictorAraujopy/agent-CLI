"""Generation through an API: ApiModel, the Model that sends the conversation to a model on OpenRouter."""

from typing import Any

import httpx

from agent_cli.inference.chat_completions import ChatCompletionsModel, connection_failure, error_in_the_body
from agent_cli.inference.model import ContextFullError


def open_client(api_key: str) -> httpx.Client:
    """Return the HTTP client that reaches OpenRouter, sending the key of the account with every request."""
    return httpx.Client(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10),
    )


def explanation_in(error: dict[str, Any]) -> str:
    """Return what an error OpenRouter reported says: its message, and the words of the provider when it has them."""
    message = error.get("message", "no reason given")
    metadata = error.get("metadata") or {}
    said_by_the_provider = metadata.get("raw")
    if said_by_the_provider is None:
        return message
    return f"{message}: {said_by_the_provider}"


def says_context_is_full(error: dict[str, Any]) -> bool:
    """Tell whether an error says the conversation does not fit: by its type, or by its text when it has no type."""
    metadata = error.get("metadata") or {}
    typed_as_full = metadata.get("error_type") == "context_length_exceeded"
    described_as_full = "maximum context length" in explanation_in(error)
    return typed_as_full or described_as_full


def failure_reported(error: dict[str, Any]) -> Exception:
    """Return the exception for an error OpenRouter reported: ContextFullError when the conversation does not fit."""
    explanation = explanation_in(error)
    if says_context_is_full(error):
        return ContextFullError(explanation)
    return RuntimeError(f"OpenRouter reported an error: {explanation}")


def endpoints_that_keep_no_data(client: httpx.Client) -> list[dict[str, Any]]:
    """Ask OpenRouter for the endpoints, of every model, whose providers keep nothing of what they receive."""
    try:
        answer = client.get("/endpoints/zdr")
    except httpx.TransportError as error:
        raise connection_failure("OpenRouter", error) from error
    if answer.status_code != 200:
        raise failure_reported(error_in_the_body(answer))
    return answer.json()["data"]


def context_tokens_of(model_name: str, endpoints: list[dict[str, Any]]) -> int:
    """Return the size of a model's context: the smallest among its endpoints, as OpenRouter picks one per request."""
    endpoints_of_the_model = [endpoint for endpoint in endpoints if endpoint["model_id"] == model_name]
    sizes = [endpoint["context_length"] for endpoint in endpoints_of_the_model if endpoint["context_length"]]
    if not sizes:
        raise ValueError(
            f"no provider on OpenRouter serves {model_name} without keeping data: check the name of the model"
        )
    return min(sizes)


def open_api_model(client: httpx.Client, model_name: str) -> "ApiModel":
    """Return the model of that name on OpenRouter, with the context size OpenRouter reports for it now."""
    endpoints = endpoints_that_keep_no_data(client)
    return ApiModel(client, model_name, context_tokens_of(model_name, endpoints))


class ApiModel(ChatCompletionsModel):
    """A model that runs on the providers of OpenRouter: each reply is one request, answered as a stream."""

    server_name = "OpenRouter"
    thinking_field = "reasoning"

    def __init__(self, client: httpx.Client, model_name: str, context_tokens: int) -> None:
        """Keep the client that reaches OpenRouter, the name of the model there and the size of its context."""
        super().__init__(client, context_tokens)
        self.model_name = model_name
        self.thinking_effort = "max"

    def fields_of_this_server(self) -> dict[str, Any]:
        """Return what OpenRouter needs besides the conversation: the model, the providers and the thinking effort."""
        # A provider slower than 40 tokens a second in more than 1 of 10 requests of the last 5 minutes goes last.
        fast_providers_first = {"p90": 40}
        return {
            "model": self.model_name,
            "provider": {"zdr": True, "data_collection": "deny", "preferred_min_throughput": fast_providers_first},
            "reasoning": {"effort": self.thinking_effort},
        }

    def failure_from(self, error: dict[str, Any]) -> Exception:
        """Return the exception for an error OpenRouter reported: ContextFullError when the conversation does not fit."""
        return failure_reported(error)

    def tokens_counted_in(self, chunk: dict[str, Any]) -> int | None:
        """Return the tokens OpenRouter counted, which only the last chunk of a reply that reached its end carries."""
        usage = chunk.get("usage")
        if usage is None:
            return None
        return usage["total_tokens"]

    def thinking_options(self) -> tuple[str, ...]:
        """Return the choices for the thinking of a model on OpenRouter: its efforts, from none to the most."""
        return ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    def thinking(self) -> str:
        """Return the effort the model is asked to think with before the replies that follow."""
        return self.thinking_effort

    def set_thinking(self, option: str) -> None:
        """Choose the effort the model thinks with before the replies that follow: one of OpenRouter's."""
        self.thinking_effort = option

    def close(self) -> None:
        """Free nothing: the model runs on OpenRouter, and the client that reaches it serves other models too."""
