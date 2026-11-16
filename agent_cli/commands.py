"""The commands of a session: what the user types starting with a slash, and what each one does."""

from collections.abc import Callable
from dataclasses import dataclass

from agent_cli.inference.catalog import Catalog


@dataclass(frozen=True)
class Outcome:
    """What a command did, for the screen: the text to show, whether it failed, and a model to open."""

    text: str
    failed: bool = False
    model_to_open: str | None = None


@dataclass(frozen=True)
class Command:
    """One command: what runs it with what was typed after its name, and the choices it takes now."""

    run: Callable[[str, Catalog], Outcome]
    choices: Callable[[Catalog], list[str]]


def is_command(text: str) -> bool:
    """Tell whether what the user typed is a command: it starts with a slash that has no space after it."""
    return text.startswith("/") and not text.startswith("/ ")


def list_of_models(catalog: Catalog) -> str:
    """Return the models available, one numbered line each, with the one in use marked."""
    lines = ["models   /model <number or name> switches"]
    for number, name in enumerate(catalog.names(), start=1):
        mark = "  (in use)" if name == catalog.name_in_use else ""
        lines.append(f"  {number}  {name}{mark}")
    return "\n".join(lines)


def name_chosen(choice: str, names: list[str]) -> str | None:
    """Return the name of the model chosen by its number in the list or by its name, or None when there is none."""
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        return names[int(choice) - 1]
    if choice in names:
        return choice
    return None


def model_choices(catalog: Catalog) -> list[str]:
    """Return what /model takes: the number of each model in the list, and its name."""
    numbers = [str(number) for number in range(1, len(catalog.names()) + 1)]
    return numbers + catalog.names()


def choose_model(choice: str, catalog: Catalog) -> Outcome:
    """Ask the screen to open the model chosen, or list the models when none is given."""
    if not choice:
        return Outcome(list_of_models(catalog))
    name = name_chosen(choice, catalog.names())
    if name is None:
        return Outcome(f"no such model: {choice}\n{list_of_models(catalog)}", failed=True)
    if name == catalog.name_in_use:
        return Outcome(f"model: {name} is already in use")
    return Outcome(f"opening {name}…", model_to_open=name)


def thinking_choices(catalog: Catalog) -> list[str]:
    """Return what /think takes: the thinking options of the model in use."""
    return list(catalog.in_use().thinking_options())


def choose_thinking(option: str, catalog: Catalog) -> Outcome:
    """Choose how much the model in use thinks, or tell the choice in use and the others when none is given."""
    model = catalog.in_use()
    options = ", ".join(model.thinking_options())
    if not option:
        return Outcome(f"thinking: {model.thinking()}   options: {options}")
    if option not in model.thinking_options():
        return Outcome(f"no such thinking option: {option}   options: {options}", failed=True)
    catalog.set_thinking(option)
    return Outcome(f"thinking: {option}")


COMMANDS = {
    "model": Command(run=choose_model, choices=model_choices),
    "think": Command(run=choose_thinking, choices=thinking_choices),
}


def completions(catalog: Catalog) -> list[str]:
    """Return every command the user can type now: each one alone, then with each choice it takes."""
    lines = [f"/{name}" for name in COMMANDS]
    for name, command in COMMANDS.items():
        lines += [f"/{name} {choice}" for choice in command.choices(catalog)]
    return lines


def run_command(text: str, catalog: Catalog) -> Outcome:
    """Run the command the user typed and return what the screen has to show and do."""
    name, _, argument = text.removeprefix("/").partition(" ")
    if name not in COMMANDS:
        commands = ", ".join(f"/{command_name}" for command_name in COMMANDS)
        return Outcome(f"no such command: /{name}   commands: {commands}", failed=True)
    return COMMANDS[name].run(argument.strip(), catalog)
