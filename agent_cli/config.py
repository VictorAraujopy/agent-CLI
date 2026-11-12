"""Configuration: reads the AGENT_CLI_ variables. It is the only module that reads the environment."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_CONTEXT_TOKENS = 8192
PROJECT_FOLDER = Path(__file__).parent.parent


@dataclass(frozen=True)
class Settings:
    """What the user configured. An empty variable becomes None, an empty tuple or, for the context, its default."""

    model_path: Path | None
    rpc_servers: tuple[str, ...]
    api_base_url: str | None
    api_key: str | None = field(repr=False)
    api_models: tuple[str, ...]
    context_tokens: int
    system_prompt_path: Path | None


def read_variables(env_file: Path, environment: Mapping[str, str]) -> dict[str, str]:
    """Return the variables of the .env file, replaced by the environment where both set one."""
    from_file = dotenv_values(env_file)
    from_file_with_value = {name: value for name, value in from_file.items() if value is not None}
    return {**from_file_with_value, **environment}


def text_or_none(variables: Mapping[str, str], name: str) -> str | None:
    """Return a variable without surrounding spaces, or None when it is empty or missing."""
    value = variables.get(name, "").strip()
    return value or None


def path_or_none(variables: Mapping[str, str], name: str) -> Path | None:
    """Return a variable as a path, or None when it is empty or missing."""
    value = text_or_none(variables, name)
    if value is None:
        return None
    # ~ becomes the home folder, and a relative path starts at the project folder, where the .env is.
    return PROJECT_FOLDER / Path(value).expanduser()


def items_of(variables: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Return the comma-separated items of a variable, without spaces and without empty items."""
    value = variables.get(name, "")
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def whole_number_or_default(variables: Mapping[str, str], name: str, default: int) -> int:
    """Return a variable as a whole number above zero, or the default when it is empty or missing."""
    value = text_or_none(variables, name)
    if value is None:
        return default
    if not value.isdecimal() or int(value) == 0:
        raise ValueError(f"{name} must be a whole number above zero, but it is {value!r}")
    return int(value)


def build_settings(variables: Mapping[str, str]) -> Settings:
    """Build the settings from the variables, giving each value its type."""
    return Settings(
        model_path=path_or_none(variables, "AGENT_CLI_MODEL_PATH"),
        rpc_servers=items_of(variables, "AGENT_CLI_RPC_SERVERS"),
        api_base_url=text_or_none(variables, "AGENT_CLI_API_BASE_URL"),
        api_key=text_or_none(variables, "AGENT_CLI_API_KEY"),
        api_models=items_of(variables, "AGENT_CLI_API_MODELS"),
        context_tokens=whole_number_or_default(
            variables, "AGENT_CLI_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS
        ),
        system_prompt_path=path_or_none(variables, "AGENT_CLI_SYSTEM_PROMPT_PATH"),
    )


def load_settings() -> Settings:
    """Read the .env file of the project folder and the environment, and return the settings."""
    env_file = PROJECT_FOLDER / ".env"
    try:
        return build_settings(read_variables(env_file, os.environ))
    except Exception as error:
        message = str(error)
    # Raised from here, after the error above is gone, on purpose. The functions below hold every
    # variable in clear, the API key included, and neither a traceback that prints local values nor
    # anything that keeps the error may reach them.
    raise ValueError(message)
