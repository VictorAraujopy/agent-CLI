"""Tests for the embedding model on OpenRouter. No test reaches the network: a stand-in answers the requests."""

import json
from collections.abc import Callable

import httpx
import pytest

from agent_cli.rag.embeddings import EmbeddingModel

OPENROUTER_API = "https://openrouter.ai/api/v1"


def answer_with_vectors(vectors: list[list[float]]) -> httpx.Response:
    """Return the answer OpenRouter gives to a request of embeddings, one item per text, in the order asked."""
    items = [{"object": "embedding", "embedding": vector, "index": index} for index, vector in enumerate(vectors)]
    return httpx.Response(200, json={"object": "list", "data": items, "model": "author/embedding"})


def embedding_model_answered_by(handler: Callable[[httpx.Request], httpx.Response]) -> EmbeddingModel:
    """Return the embedding model with a client whose requests the handler answers instead of OpenRouter."""
    client = httpx.Client(base_url=OPENROUTER_API, transport=httpx.MockTransport(handler))
    return EmbeddingModel(client, "author/embedding")


def test_texts_go_to_the_embedding_model_on_providers_that_keep_nothing() -> None:
    """The request names the model, carries the texts, and allows only providers that keep no data."""
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        """Keep the request and answer with one vector per text."""
        requests.append(request)
        return answer_with_vectors([[1.0, 0.0], [0.0, 1.0]])

    vectors = embedding_model_answered_by(answer).vectors_of(["first", "second"])
    [request] = requests
    assert (request.method, request.url.path) == ("POST", "/api/v1/embeddings")
    assert json.loads(request.content) == {
        "model": "author/embedding",
        "input": ["first", "second"],
        "provider": {"zdr": True, "data_collection": "deny"},
    }
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_vectors_follow_the_index_of_their_items_whatever_order_the_items_come_in() -> None:
    """Each vector goes with the text at its index, even when OpenRouter lists the items in another order."""

    def answer(request: httpx.Request) -> httpx.Response:
        """Answer with the items of the texts listed last first."""
        items = [{"embedding": [2.0], "index": 1}, {"embedding": [1.0], "index": 0}]
        return httpx.Response(200, json={"data": items})

    assert embedding_model_answered_by(answer).vectors_of(["first", "second"]) == [[1.0], [2.0]]


def test_refusal_of_openrouter_ends_with_its_reason() -> None:
    """A request OpenRouter refuses, such as one naming no embedding model it has, fails with what it said."""
    refusal = {"error": {"message": "No such embedding model.", "code": 400}}
    embedding_model = embedding_model_answered_by(lambda request: httpx.Response(400, json=refusal))
    with pytest.raises(RuntimeError, match="No such embedding model."):
        embedding_model.vectors_of(["first"])


def test_refusal_with_no_error_in_its_body_ends_with_its_status() -> None:
    """A refusal whose body is not an error of OpenRouter, such as the page of a gateway, fails with its HTTP status."""
    page = httpx.Response(502, text="<html>Bad Gateway</html>")
    embedding_model = embedding_model_answered_by(lambda request: page)
    with pytest.raises(RuntimeError, match="HTTP 502 Bad Gateway"):
        embedding_model.vectors_of(["first"])


@pytest.mark.parametrize(
    ("vectors", "reason"),
    [([[1.0]], "1 vectors for 2 texts"), ([[1.0], []], "an empty vector")],
)
def test_answer_without_a_vector_for_each_text_is_an_error(vectors: list[list[float]], reason: str) -> None:
    """A vector missing or empty would be kept on the wrong turn or break every search, so the answer is refused."""
    embedding_model = embedding_model_answered_by(lambda request: answer_with_vectors(vectors))
    with pytest.raises(RuntimeError, match=reason):
        embedding_model.vectors_of(["first", "second"])


def test_error_in_the_body_of_a_200_ends_with_its_reason() -> None:
    """A provider that fails after OpenRouter answered 200 leaves only an error, which fails the request."""
    error = {"error": {"message": "Provider returned error", "code": 502, "metadata": {"raw": "overloaded"}}}
    embedding_model = embedding_model_answered_by(lambda request: httpx.Response(200, json=error))
    with pytest.raises(RuntimeError, match="Provider returned error: overloaded"):
        embedding_model.vectors_of(["first"])


def test_network_that_fails_ends_with_a_connection_error() -> None:
    """With no way to reach OpenRouter, the request fails as a connection error that names it."""

    def fail(request: httpx.Request) -> httpx.Response:
        """Fail as a network that is down would."""
        raise httpx.ConnectError("no route to host")

    with pytest.raises(ConnectionError, match="the connection to OpenRouter failed: no route to host"):
        embedding_model_answered_by(fail).vectors_of(["first"])
