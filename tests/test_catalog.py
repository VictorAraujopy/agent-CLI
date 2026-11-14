"""Tests for the catalog of the models a session can use."""

import pytest

from agent_cli.inference.catalog import Catalog
from test_app import ScriptedModel


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
    catalog.open("first")
    catalog.remember_thinking("off")
    assert catalog.open("second").thinking() == "on"
    assert catalog.open("first").thinking() == "off"
