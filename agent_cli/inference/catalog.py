"""The models a session can use: the catalog opens the one chosen and closes the one it replaces."""

from collections.abc import Callable

from agent_cli.inference.model import Model


class Catalog:
    """The models available to a session, by name, and the one in use."""

    def __init__(self, how_to_open: dict[str, Callable[[], Model]]) -> None:
        """Keep, for the name of each model, the function that opens it, with no model in use yet."""
        self.how_to_open = how_to_open
        self.name_in_use: str | None = None
        self.model_in_use: Model | None = None
        self.thinking_chosen: dict[str, str] = {}

    def names(self) -> list[str]:
        """Return the names of the models available, in the order they were configured."""
        return list(self.how_to_open)

    def in_use(self) -> Model:
        """Return the model in use; asking before the first one was opened is an error."""
        if self.model_in_use is None:
            raise RuntimeError("no model was opened yet")
        return self.model_in_use

    def open(self, name: str) -> Model:
        """Open the named model and put it in use, closing the one that was; a failure leaves that one in use."""
        model = self.how_to_open[name]()
        if name in self.thinking_chosen:
            model.set_thinking(self.thinking_chosen[name])
        if self.model_in_use is not None:
            self.model_in_use.close()
        self.name_in_use = name
        self.model_in_use = model
        return model

    def set_thinking(self, option: str) -> None:
        """Choose the thinking of the model in use, and remember it for when that model is opened again."""
        self.in_use().set_thinking(option)
        if self.name_in_use is not None:
            self.thinking_chosen[self.name_in_use] = option
