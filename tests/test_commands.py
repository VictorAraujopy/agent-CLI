"""Tests for the commands of a session."""

import pytest

from agent_cli.commands import Outcome, completions, is_command, run_command
from agent_cli.inference.catalog import Catalog
from test_app import ScriptedModel


def catalog_of_two() -> Catalog:
    """Return a catalog with two models, the first one in use."""
    catalog = Catalog({"local": lambda: ScriptedModel([]), "author/model": lambda: ScriptedModel([])})
    catalog.open("local")
    return catalog


@pytest.mark.parametrize(
    ("text", "command"),
    [("/think", True), ("/think off", True), ("/", True), ("/ think", False), ("a /think", False), ("think", False)],
)
def test_only_a_slash_at_the_start_with_no_space_after_it_makes_a_command(text: str, command: bool) -> None:
    """A slash followed by a space, or anywhere but the start, is text for the model."""
    assert is_command(text) is command


def test_think_alone_tells_the_choice_in_use_and_the_others() -> None:
    """With no option the command changes nothing and shows what can be chosen."""
    catalog = catalog_of_two()
    assert run_command("/think", catalog.model_in_use, catalog) == Outcome("thinking: on   options: on, off")
    assert catalog.model_in_use.thinking() == "on"


def test_think_with_an_option_sets_it_on_the_model_and_remembers_it() -> None:
    """The option goes to the model in use and is kept for when that model is opened again."""
    catalog = catalog_of_two()
    assert run_command("/think  off ", catalog.model_in_use, catalog) == Outcome("thinking: off")
    assert catalog.model_in_use.thinking() == "off"
    assert catalog.thinking_chosen == {"local": "off"}


def test_think_with_an_option_the_model_does_not_have_fails_and_changes_nothing() -> None:
    """An effort is not a choice of a model whose thinking is only on or off."""
    catalog = catalog_of_two()
    outcome = run_command("/think high", catalog.model_in_use, catalog)
    assert outcome == Outcome("no such thinking option: high   options: on, off", failed=True)
    assert catalog.model_in_use.thinking() == "on"


def test_model_alone_lists_the_models_with_the_one_in_use_marked() -> None:
    """With no choice the command opens nothing and shows the numbered list."""
    catalog = catalog_of_two()
    listed = "models   /model <number or name> switches\n  1  local  (in use)\n  2  author/model"
    assert run_command("/model", catalog.model_in_use, catalog) == Outcome(listed)


@pytest.mark.parametrize("choice", ["2", "author/model"])
def test_model_chosen_by_number_or_by_name_is_handed_to_the_screen_to_open(choice: str) -> None:
    """The command opens nothing itself: opening takes time, so the screen does it in the background."""
    catalog = catalog_of_two()
    outcome = run_command(f"/model {choice}", catalog.model_in_use, catalog)
    assert outcome == Outcome("opening author/model…", model_to_open="author/model")
    assert catalog.name_in_use == "local"


def test_model_already_in_use_is_not_opened_again() -> None:
    """Choosing the model in use only says so."""
    catalog = catalog_of_two()
    assert run_command("/model 1", catalog.model_in_use, catalog) == Outcome("model: local is already in use")


@pytest.mark.parametrize("choice", ["0", "3", "other"])
def test_model_that_is_not_in_the_list_fails_and_shows_the_list(choice: str) -> None:
    """A number out of the list or a name that is not in it opens nothing."""
    catalog = catalog_of_two()
    outcome = run_command(f"/model {choice}", catalog.model_in_use, catalog)
    assert outcome.failed and outcome.model_to_open is None
    assert outcome.text.startswith(f"no such model: {choice}\nmodels")


def test_unknown_command_fails_and_names_the_commands() -> None:
    """Anything else after the slash is not a command of the session."""
    catalog = catalog_of_two()
    outcome = run_command("/help me", catalog.model_in_use, catalog)
    assert outcome == Outcome("no such command: /help   commands: /model, /think", failed=True)


def test_completions_are_the_commands_with_every_choice_they_take_now() -> None:
    """The message box suggests the commands, the models by number and name, and the options of the model in use."""
    catalog = catalog_of_two()
    assert completions(catalog.model_in_use, catalog) == [
        "/model", "/think", "/model 1", "/model 2", "/model local", "/model author/model", "/think on", "/think off",
    ]
