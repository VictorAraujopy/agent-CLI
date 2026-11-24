"""The embedding model on OpenRouter: turns texts into vectors, on providers that keep nothing of them."""

from typing import Any

import httpx

from agent_cli.inference.api import failure_reported
from agent_cli.inference.chat_completions import connection_failure, error_in_the_body


def vectors_in(items: list[dict[str, Any]], texts: list[str]) -> list[list[float]]:
    """Return the vector of each text from the items of the answer, by their index; one missing or empty is an error."""
    vectors = [item["embedding"] for item in sorted(items, key=lambda item: item["index"])]
    if len(vectors) != len(texts):
        raise RuntimeError(f"OpenRouter returned {len(vectors)} vectors for {len(texts)} texts")
    if not all(vectors):
        raise RuntimeError("OpenRouter returned an empty vector")
    return vectors


class EmbeddingModel:
    """A model on OpenRouter that turns a text into a vector: texts of close meaning get vectors of close direction."""

    def __init__(self, client: httpx.Client, model_name: str) -> None:
        """Keep the client that reaches OpenRouter and the name of the embedding model there."""
        self.client = client
        self.model_name = model_name

    def vectors_of(self, texts: list[str]) -> list[list[float]]:
        """Ask OpenRouter, in one request, for the vector of each text, made by a provider that keeps nothing."""
        request = {"model": self.model_name, "input": texts, "provider": {"zdr": True, "data_collection": "deny"}}
        try:
            answer = self.client.post("/embeddings", json=request)
        except httpx.TransportError as error:
            raise connection_failure("OpenRouter", error) from error
        if answer.status_code != 200:
            raise failure_reported(error_in_the_body(answer))
        body = answer.json()
        # A provider that fails after OpenRouter answered 200 leaves only an error in the body.
        if "error" in body:
            raise failure_reported(body["error"])
        return vectors_in(body["data"], texts)
