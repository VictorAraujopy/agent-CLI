"""Tests for the catalog of the models a session can use."""

import pytest

from agent_cli.inference.catalog import Catalog
from scripted_model import ScriptedModel


def test_names_come_in_the_order_the_models_were_configured() -> None:
    """The first name is the model the session starts on."""
    catalog = Catalog({"local": lambda: ScriptedModel([]), "author/model": lambda: ScriptedModel([])})
    assert catalog.names() == ["local", "author/model"]
    assert catalog.name_in_use is None


def test_opened_model_is_in_use_and_the_one_it_replaces_is_closed() -> None:
    """Opening a model puts it in use; the one that was in use is closed only then."""
    first, second = ScriptedModel([]), ScriptedModel([])
    catalog = Catalog({"first": lambda: first, "second": lambda: second})
    assert catalog.open("first") is first
    assert not first.was_closed
    assert catalog.open("second") is second
    assert (catalog.name_in_use, catalog.model_in_use) == ("second", second)
    assert first.was_closed and not second.was_closed


def test_model_that_fails_to_open_leaves_the_one_in_use_open() -> None:
    """A failure to open the new model closes nothing and changes nothing."""
    first = ScriptedModel([])

    def fail_to_open() -> ScriptedModel:
        """Stand in for a model that cannot be opened."""
        raise ConnectionError("the server did not answer")

    catalog = Catalog({"first": lambda: first, "second": fail_to_open})
    catalog.open("first")
    with pytest.raises(ConnectionError):
        catalog.open("second")
    assert (catalog.name_in_use, catalog.model_in_use) == ("first", first)
    assert not first.was_closed


def test_thinking_chosen_for_a_model_is_given_back_when_it_is_opened_again() -> None:
    """Each model keeps its own choice during the session; one never chosen for starts as the model starts."""
    catalog = Catalog({"first": lambda: ScriptedModel([]), "second": lambda: ScriptedModel([])})
    first = catalog.open("first")
    catalog.set_thinking("off")
    assert first.thinking() == "off"
    assert catalog.open("second").thinking() == "on"
    assert catalog.open("first").thinking() == "off"


def test_asking_for_the_model_in_use_before_any_was_opened_is_an_error() -> None:
    """The catalog has no model in use until the first one is opened, and then hands it over."""
    model = ScriptedModel([])
    catalog = Catalog({"first": lambda: model})
    with pytest.raises(RuntimeError, match="no model was opened yet"):
        catalog.in_use()
    catalog.open("first")
    assert catalog.in_use() is model
