"""The search of earlier turns: finds those closest in meaning to the user's message, for the model to read with it."""

import threading
from itertools import batched

from agent_cli.history.queries import History, SavedExcerpt, TurnToIndex
from agent_cli.inference.model import Message, thinking_and_answer
from agent_cli.rag.embeddings import EmbeddingModel

CHARACTERS_OF_AN_EXCERPT = 1500
# Ten texts a request: a request that fails loses little, and a stop is seen between two of them.
TEXTS_PER_REQUEST = 10


def excerpt_of(messages: list[Message]) -> str:
    """Return the text a turn is found by and brought back as: the user's message and the model's last answer."""
    user_text = messages[0]["content"]
    replies = [message["content"] for message in messages if message["role"] == "assistant"]
    last_reply = replies[-1] if replies else ""
    _, last_answer = thinking_and_answer(last_reply)
    excerpt = f"User: {user_text}\nAssistant: {last_answer}"
    return excerpt[:CHARACTERS_OF_AN_EXCERPT]


def with_recalled(user_message: Message, excerpts: list[str]) -> Message:
    """Return the user's message as the model reads it: what the search brought back, then what the user wrote."""
    if not excerpts:
        return user_message
    turns = "".join(f"<turn>\n{excerpt}\n</turn>\n" for excerpt in excerpts)
    recalled = (
        "<recalled>\n"
        "Turns of earlier conversations in this folder, the closest in meaning to my message. "
        "They may be outdated or unrelated.\n"
        f"{turns}"
        "</recalled>\n\n"
    )
    return {**user_message, "content": recalled + user_message["content"]}


class Recall:
    """The search of the earlier turns of the folder by meaning, with their vectors kept in the history."""

    def __init__(self, embedding_model: EmbeddingModel, history: History) -> None:
        """Keep the model that makes the vectors and the history that keeps them."""
        self.embedding_model = embedding_model
        self.history = history

    def closest_excerpts(self, user_text: str, turns_in_context: int, stop_requested: threading.Event) -> list[str]:
        """Return up to three excerpts above 0.6 of similarity, closest first, leaving out the turns in the context."""
        message_vector = self.embed_message_and_index_new_turns(user_text, stop_requested)
        # The similarity is the cosine of the angle between the two vectors. At 0.6 or below a turn is taken as
        # unrelated; the scale of the cosine changes with the embedding model.
        return self.history.excerpts_closest_to(
            self.embedding_model.model_name,
            message_vector,
            turns_in_context,
            minimum_similarity=0.6,
            how_many=3,
        )

    def embed_message_and_index_new_turns(self, user_text: str, stop_requested: threading.Event) -> list[float]:
        """Return the vector of the message, indexing on the way the turns that have none, until a stop is asked."""
        turns_to_index = self.history.turns_with_no_excerpt(self.embedding_model.model_name)
        turns_with_the_message = turns_to_index[: TEXTS_PER_REQUEST - 1]
        message_vector = self.embed_message_and_index(user_text, turns_with_the_message)
        for later_turns in batched(turns_to_index[TEXTS_PER_REQUEST - 1 :], TEXTS_PER_REQUEST):
            if stop_requested.is_set():
                break
            self.index(list(later_turns))
        return message_vector

    def embed_message_and_index(self, user_text: str, turns: list[TurnToIndex]) -> list[float]:
        """Return the vector of the message, and keep those of the turns, all made in the same request."""
        texts = [excerpt_of(turn.messages) for turn in turns]
        message_text = user_text[:CHARACTERS_OF_AN_EXCERPT]
        message_vector, *turn_vectors = self.embedding_model.vectors_of([message_text, *texts])
        self.keep(turns, texts, turn_vectors)
        return message_vector

    def index(self, turns: list[TurnToIndex]) -> None:
        """Keep the vectors of the turns, made in one request."""
        texts = [excerpt_of(turn.messages) for turn in turns]
        self.keep(turns, texts, self.embedding_model.vectors_of(texts))

    def keep(self, turns: list[TurnToIndex], texts: list[str], vectors: list[list[float]]) -> None:
        """Save in the history the excerpt of each turn with its vector, so a later failure does not lose it."""
        excerpts = [
            SavedExcerpt(turn.conversation_id, turn.turn, text, vector)
            for turn, text, vector in zip(turns, texts, vectors, strict=True)
        ]
        self.history.save_excerpts(self.embedding_model.model_name, excerpts)
