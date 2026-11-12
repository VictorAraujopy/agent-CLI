"""Tests for the configuration module."""

import os
import traceback
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from agent_cli import config
from agent_cli.config import build_settings, load_settings, read_variables


def clear_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the AGENT_CLI_ variables from the environment for the length of one test."""
    for name in list(os.environ):
        if name.startswith("AGENT_CLI_"):
            monkeypatch.delenv(name)


def test_nothing_configured_gives_empty_settings() -> None:
    """Variables that are missing become None or an empty tuple."""
    settings = build_settings({})
    assert settings.model_path is None
    assert settings.rpc_servers == ()
    assert settings.api_key is None
    assert settings.api_models == ()
    assert settings.system_prompt_path is None


def test_blank_values_count_as_not_configured() -> None:
    """A variable with only spaces is the same as a missing one."""
    variables = {
        "AGENT_CLI_MODEL_PATH": "   ",
        "AGENT_CLI_API_KEY": " ",
        "AGENT_CLI_API_MODELS": " , ",
    }
    settings = build_settings(variables)
    assert settings.model_path is None
    assert settings.api_key is None
    assert settings.api_models == ()


def test_paths_lose_spaces_and_expand_the_home_folder() -> None:
    """A path is stripped, and ~ becomes the home folder."""
    settings = build_settings({"AGENT_CLI_MODEL_PATH": " ~/models/small.gguf "})
    assert settings.model_path == Path.home() / "models" / "small.gguf"


def test_lists_are_split_on_commas() -> None:
    """A list loses the spaces around each item and the empty items."""
    settings = build_settings({"AGENT_CLI_RPC_SERVERS": "10.0.0.2:50052, 10.0.0.3:50052,"})
    assert settings.rpc_servers == ("10.0.0.2:50052", "10.0.0.3:50052")


def test_api_key_is_left_out_of_the_printed_settings() -> None:
    """Printing the settings must not reveal the key."""
    settings = build_settings({"AGENT_CLI_API_KEY": "secret-value"})
    assert settings.api_key == "secret-value"
    assert "secret-value" not in repr(settings)


def test_settings_cannot_be_changed() -> None:
    """The settings are read once and stay as they were read."""
    settings = build_settings({})
    with pytest.raises(FrozenInstanceError):
        settings.api_key = "other"


def test_environment_replaces_the_env_file(tmp_path: Path) -> None:
    """When both set a variable, the environment wins and the rest of the file is kept."""
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_CLI_MODEL_PATH=/from/file.gguf\nAGENT_CLI_API_MODELS=one,two\n")
    variables = read_variables(env_file, {"AGENT_CLI_MODEL_PATH": "/from/environment.gguf"})
    assert variables["AGENT_CLI_MODEL_PATH"] == "/from/environment.gguf"
    assert variables["AGENT_CLI_API_MODELS"] == "one,two"


def test_missing_env_file_leaves_only_the_environment(tmp_path: Path) -> None:
    """Without a .env file the variables come from the environment alone."""
    variables = read_variables(tmp_path / ".env", {"AGENT_CLI_API_KEY": "from-environment"})
    assert variables == {"AGENT_CLI_API_KEY": "from-environment"}


def test_load_settings_reads_the_env_file_of_the_project_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .env file is the one next to the code, wherever the program is called from."""
    clear_variables(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nAGENT_CLI_API_MODELS=one, two\n")
    monkeypatch.setattr(config, "PROJECT_FOLDER", tmp_path)
    settings = load_settings()
    assert settings.api_models == ("one", "two")
    assert settings.model_path is None


def test_env_file_of_the_directory_the_program_is_called_from_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .env left in some other folder, such as a cloned repository, never configures the program."""
    clear_variables(monkeypatch)
    project_folder = tmp_path / "project"
    other_folder = tmp_path / "other"
    project_folder.mkdir()
    other_folder.mkdir()
    (other_folder / ".env").write_text("AGENT_CLI_RPC_SERVERS=10.0.0.9:50052\n")
    monkeypatch.setattr(config, "PROJECT_FOLDER", project_folder)
    monkeypatch.chdir(other_folder)
    assert load_settings().rpc_servers == ()


def test_configuration_error_carries_no_variable_that_could_be_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error of a wrong value keeps no function that holds the variables, the API key included."""
    clear_variables(monkeypatch)
    env_file = tmp_path / ".env"
    wrong_line = "AGENT_CLI_MODEL_PATH=~user-that-does-not-exist/model.gguf"
    env_file.write_text(f"AGENT_CLI_API_KEY=secret-value\n{wrong_line}\n")
    monkeypatch.setattr(config, "PROJECT_FOLDER", tmp_path)
    with pytest.raises(ValueError) as raised:
        load_settings()
    local_values = [repr(frame.f_locals) for frame, _ in traceback.walk_tb(raised.value.__traceback__)]
    assert "secret-value" not in "".join(local_values)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None


def test_project_folder_is_the_one_that_holds_the_package_and_the_env_example() -> None:
    """The folder the .env file is read from is the root of the project, not the package inside it."""
    assert (config.PROJECT_FOLDER / "agent_cli" / "config.py").is_file()
    assert (config.PROJECT_FOLDER / ".env.example").is_file()


def test_relative_path_starts_at_the_project_folder() -> None:
    """A path written without a root is taken from the folder of the .env file, not from the current one."""
    settings = build_settings({"AGENT_CLI_SYSTEM_PROMPT_PATH": "storage/prompt.md"})
    assert settings.system_prompt_path == config.PROJECT_FOLDER / "storage" / "prompt.md"


def test_every_variable_of_the_env_example_reaches_the_settings() -> None:
    """The example file and the code must name the same variables: none unread, none missing."""
    env_example_file = Path(__file__).parent.parent / ".env.example"
    lines = env_example_file.read_text().splitlines()
    names_in_example = [line.split("=")[0] for line in lines if line.startswith("AGENT_CLI_")]
    settings = build_settings({name: "12345" for name in names_in_example})
    values = [getattr(settings, field.name) for field in fields(settings)]
    the_filled_value_in_each_type = [config.PROJECT_FOLDER / "12345", ("12345",), "12345"]
    assert len(names_in_example) == len(values)
    assert all(value in the_filled_value_in_each_type for value in values)
