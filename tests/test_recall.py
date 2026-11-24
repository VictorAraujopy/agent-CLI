"""Tests for the search of earlier turns: the excerpt of a turn, the similarity, and what the model reads with it."""

import math
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_cli.history.queries import History
from agent_cli.rag.recall import Recall, excerpt_of, with_recalled

READ_CALL = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}


class EmbeddingModelByWord:
    """Stands in for the embedding model: a text gets the vector of the first word of the table it holds."""

    model_name = "author/embedding"

    def __init__(self, vector_of_word: dict[str, list[float]], failing_request: int | None = None) -> None:
        """Keep the vector of each word and the number of the request that fails, if any, with no request yet."""
        self.vector_of_word = vector_of_word
        self.failing_request = failing_request
        self.texts_asked: list[list[str]] = []
        self.while_asking: Callable[[], None] = lambda: None

    def vectors_of(self, texts: list[str]) -> list[list[float]]:
        """Keep the texts of the request, do what the test set for its middle, and return the vector of each text."""
        self.texts_asked.append(texts)
        self.while_asking()
        if len(self.texts_asked) == self.failing_request:
            raise ConnectionError("the connection to OpenRouter failed: reset by peer")
        return [self.vector_of(text) for text in texts]

    def vector_of(self, text: str) -> list[float]:
        """Return the vector of the first word of the table that the text holds."""
        for word, vector in self.vector_of_word.items():
            if word in text:
                return vector
        raise AssertionError(f"the test gave no vector for {text!r}")


def at_similarity(cosine: float) -> list[float]:
    """Return a vector whose similarity to [1, 0] is the cosine given."""
    return [cosine, math.sqrt(1 - cosine**2)]


def exchange(question: str, answer: str) -> list[dict[str, str]]:
    """Return the messages of a turn with no tool."""
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def search(recall: Recall, user_text: str, turns_in_context: int = 0) -> list[str]:
    """Return what the search finds for the message, in a turn nobody stops."""
    return recall.closest_excerpts(user_text, turns_in_context, threading.Event())


def history_with_turns(tmp_path: Path, questions: list[str]) -> History:
    """Return the history of a folder, from an earlier session, with one turn for each question, answered ok."""
    earlier_session = History(tmp_path / "history.db", tmp_path / "project")
    for question in questions:
        earlier_session.save_turn(exchange(question, "ok"), {}, 10)
    return History(tmp_path / "history.db", tmp_path / "project")


def test_excerpt_is_the_message_and_the_last_answer_without_its_thinking() -> None:
    """The replies that only asked for tools stay out, and so does what the last one thought."""
    messages = [
        {"role": "user", "content": "Read a"},
        {"role": "assistant", "content": "<think>\nplan\n</think>\n\nLooking.", "tool_calls": [READ_CALL]},
        {"role": "assistant", "content": "<think>\nread it\n</think>\n\nIt says hello."},
    ]
    assert excerpt_of(messages) == "User: Read a\nAssistant: It says hello."


def test_excerpt_is_cut_at_1500_characters() -> None:
    """A long turn keeps only its start, so a few excerpts fit in one request and in the context."""
    excerpt = excerpt_of(exchange("x" * 1000, "y" * 1000))
    assert excerpt == "User: " + "x" * 1000 + "\nAssistant: " + "y" * 482
    assert len(excerpt) == 1500


def test_turn_with_no_reply_is_found_by_its_message_alone() -> None:
    """A turn kept with no reply of the model still gets an excerpt, instead of failing every search of the folder."""
    assert excerpt_of([{"role": "user", "content": "Hi"}]) == "User: Hi\nAssistant: "


def test_message_with_nothing_recalled_goes_as_the_user_wrote_it() -> None:
    """When the search brings nothing back, the model reads only what the user wrote."""
    message = {"role": "user", "content": "Hi"}
    assert with_recalled(message, []) == {"role": "user", "content": "Hi"}


def test_recalled_turns_come_before_the_message_each_in_its_tag() -> None:
    """The model reads what was recalled first, each turn apart, and what the user wrote last."""
    excerpts = ["User: a\nAssistant: b", "User: c\nAssistant: d"]
    recalled = with_recalled({"role": "user", "content": "And now?"}, excerpts)
    assert recalled == {
        "role": "user",
        "content": "<recalled>\n"
        "Turns of earlier conversations in this folder, the closest in meaning to my message. "
        "They may be outdated or unrelated.\n"
        "<turn>\nUser: a\nAssistant: b\n</turn>\n"
        "<turn>\nUser: c\nAssistant: d\n</turn>\n"
        "</recalled>\n\n"
        "And now?",
    }


def test_turns_with_no_vector_are_indexed_in_the_request_of_the_message(tmp_path: Path) -> None:
    """The first search sends the message and every turn not yet indexed; the next sends the message alone."""
    history = history_with_turns(tmp_path, ["python lists", "python dicts"])
    embedding_model = EmbeddingModelByWord({"python": [1.0, 0.0]})
    recall = Recall(embedding_model, history)
    found = search(recall, "python sorting")
    assert embedding_model.texts_asked == [
        ["python sorting", "User: python lists\nAssistant: ok", "User: python dicts\nAssistant: ok"]
    ]
    assert sorted(found) == ["User: python dicts\nAssistant: ok", "User: python lists\nAssistant: ok"]
    search(recall, "python again")
    assert embedding_model.texts_asked[1] == ["python again"]


def test_only_the_three_closest_turns_above_0_6_come_back_the_closest_first(tmp_path: Path) -> None:
    """Of the turns closer than 0.6 to the message, the three closest come back, from the closest to the farthest."""
    history = history_with_turns(tmp_path, ["at 0.7", "at 0.95", "at 0.59", "at 0.8", "at 0.61", "at 0.9"])
    vector_of_word = {"message": [1.0, 0.0]}
    for cosine in ("0.7", "0.95", "0.59", "0.8", "0.61", "0.9"):
        vector_of_word[f"at {cosine}"] = at_similarity(float(cosine))
    recall = Recall(EmbeddingModelByWord(vector_of_word), history)
    assert search(recall, "message") == [
        "User: at 0.95\nAssistant: ok",
        "User: at 0.9\nAssistant: ok",
        "User: at 0.8\nAssistant: ok",
    ]


def test_turn_at_0_6_or_below_never_comes_back(tmp_path: Path) -> None:
    """A turn not closer than 0.6 is left out even when it is the closest there is."""
    history = history_with_turns(tmp_path, ["at 0.61", "at 0.59"])
    vector_of_word = {"message": [1.0, 0.0], "at 0.61": at_similarity(0.61), "at 0.59": at_similarity(0.59)}
    recall = Recall(EmbeddingModelByWord(vector_of_word), history)
    assert search(recall, "message") == ["User: at 0.61\nAssistant: ok"]
    history_far = history_with_turns(tmp_path / "far", ["at 0.59"])
    recall_far = Recall(EmbeddingModelByWord(vector_of_word), history_far)
    assert search(recall_far, "message") == []


def test_turns_the_model_still_reads_are_not_recalled(tmp_path: Path) -> None:
    """The last turns of the conversation of the session are already in the context, so only older ones come back."""
    history = History(tmp_path / "history.db", tmp_path / "project")
    history.save_turn(exchange("python one", "ok"), {}, 10)
    history.save_turn(exchange("python two", "ok"), {}, 10)
    recall = Recall(EmbeddingModelByWord({"python": [1.0, 0.0]}), history)
    assert search(recall, "python three", turns_in_context=1) == ["User: python one\nAssistant: ok"]


def test_long_message_is_searched_by_its_first_1500_characters(tmp_path: Path) -> None:
    """The message is cut as the excerpts are, so a pasted file does not pass the limit of the embedding model."""
    embedding_model = EmbeddingModelByWord({"python": [1.0, 0.0]})
    recall = Recall(embedding_model, History(tmp_path / "history.db", tmp_path / "project"))
    search(recall, "python " + "x" * 3000)
    assert embedding_model.texts_asked == [["python " + "x" * 1493]]


def excerpts_of_turns(numbers: range) -> list[str]:
    """Return the excerpts of the turns saved by history_with_turns for the questions "python <number>"."""
    return [f"User: python {number}\nAssistant: ok" for number in numbers]


def test_turns_of_a_long_history_are_indexed_ten_texts_a_request(tmp_path: Path) -> None:
    """The message goes with the first nine turns, and the rest of the turns go ten to a request."""
    history = history_with_turns(tmp_path, [f"python {number}" for number in range(25)])
    embedding_model = EmbeddingModelByWord({"python": [1.0, 0.0]})
    search(Recall(embedding_model, history), "python sorting")
    assert embedding_model.texts_asked == [
        ["python sorting", *excerpts_of_turns(range(0, 9))],
        excerpts_of_turns(range(9, 19)),
        excerpts_of_turns(range(19, 25)),
    ]


def test_turns_indexed_before_a_request_that_fails_are_kept(tmp_path: Path) -> None:
    """A failed request ends the search, but the turns of the requests before it are not sent again."""
    history = history_with_turns(tmp_path, [f"python {number}" for number in range(25)])
    failing_once = EmbeddingModelByWord({"python": [1.0, 0.0]}, failing_request=2)
    with pytest.raises(ConnectionError):
        search(Recall(failing_once, history), "python sorting")
    embedding_model = EmbeddingModelByWord({"python": [1.0, 0.0]})
    search(Recall(embedding_model, history), "python again")
    assert embedding_model.texts_asked[0] == ["python again", *excerpts_of_turns(range(9, 18))]


def test_stop_during_the_indexing_ends_it_between_two_requests(tmp_path: Path) -> None:
    """A stop asked while a request runs sends no other, and the search still answers with what is indexed."""
    history = history_with_turns(tmp_path, [f"python {number}" for number in range(25)])
    embedding_model = EmbeddingModelByWord({"python": [1.0, 0.0]})
    stop_requested = threading.Event()
    embedding_model.while_asking = stop_requested.set
    found = Recall(embedding_model, history).closest_excerpts("python sorting", 0, stop_requested)
    assert len(embedding_model.texts_asked) == 1
    assert len(found) == 3
    assert set(found) <= set(excerpts_of_turns(range(0, 9)))
