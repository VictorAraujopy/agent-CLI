"""Tests for the memory the loader measures, and for the context and the options it loads a model with."""

import ctypes
from pathlib import Path
from typing import Any

import llama_cpp
import pytest

from agent_cli.inference import loader
from agent_cli.inference.loader import (
    LARGER_EXTRA_CONTEXT,
    PROBE_CONTEXT,
    SMALLER_EXTRA_CONTEXT,
    DeviceMemory,
    confirm_devices_hold,
    device_memory,
    free_memory_by_device,
    load_model,
    measure_memory_use,
    memory_another_context_takes,
)

SMALL_MODEL_TRAINED_CONTEXT = 2048
BYTES_A_MEASUREMENT_MAY_BE_OFF = 1024**2


class LibraryWithOneDevice:
    """Stands in for llama.cpp: one device, with the free and total memory the test gives, as llama.cpp counts them."""

    def __init__(self, free_bytes: int, bytes_in_all: int) -> None:
        """Keep the two counts the device will report, as unsigned numbers."""
        self.free_bytes = free_bytes
        self.bytes_in_all = bytes_in_all

    def ggml_backend_dev_count(self) -> int:
        """Tell that there is one device."""
        return 1

    def ggml_backend_dev_get(self, index: int) -> int:
        """Return the index itself as the device."""
        return index

    def ggml_backend_dev_memory(self, device: int, free_bytes: Any, bytes_in_all: Any) -> None:
        """Write the two counts where llama.cpp would, in the numbers passed by reference."""
        free_bytes._obj.value = self.free_bytes
        bytes_in_all._obj.value = self.bytes_in_all

    def ggml_backend_dev_name(self, device: int) -> bytes:
        """Return the name of the device."""
        return b"GPU0"

    def ggml_backend_dev_description(self, device: int) -> bytes:
        """Return the description of the device."""
        return b"test card"


class ModelThatRecordsItsClosing:
    """Stands in for a loaded model: it only remembers whether it was closed."""

    def __init__(self) -> None:
        """Start open."""
        self.was_closed = False

    def close(self) -> None:
        """Remember the closing."""
        self.was_closed = True


def skip_without_a_gpu() -> None:
    """Skip the test where llama.cpp has no GPU: only a GPU reports the memory a model takes of it."""
    if not llama_cpp.llama_supports_gpu_offload():
        pytest.skip("llama.cpp has no GPU on this machine, so no device reports the memory a model takes")


def hand_messages_back_to_the_bindings() -> None:
    """Undo the silencing an earlier test left in place: the bindings get the messages of llama.cpp and print them."""
    llama_cpp.llama_log_set(llama_cpp._logger.llama_log_callback, ctypes.c_void_p(0))


def test_measurements_of_a_device_become_what_the_model_takes_of_it() -> None:
    """Weights of 70000 bytes, 5000 for any context and 100 for each token are found back from four measurements."""
    weights, any_context, each_token, free_before = 70_000, 5_000, 100, 1_000_000
    free_with_the_probe = free_before - weights - any_context - each_token * PROBE_CONTEXT
    smaller_context = any_context + each_token * SMALLER_EXTRA_CONTEXT
    larger_context = any_context + each_token * LARGER_EXTRA_CONTEXT
    measured = device_memory("GPU0 (test card)", free_before, free_with_the_probe, smaller_context, larger_context)
    assert measured == DeviceMemory("GPU0 (test card)", free_before, weights, any_context, float(each_token))


def test_free_memory_is_told_for_each_device_by_name_and_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """The free bytes llama.cpp reports for a device are returned under the device's name and description."""
    monkeypatch.setattr(loader, "llama_cpp_library", lambda: LibraryWithOneDevice(free_bytes=600, bytes_in_all=1000))
    assert free_memory_by_device() == {"GPU0 (test card)": 600}


def test_device_holding_more_than_it_has_is_told_as_negative_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device 500 bytes over its memory reports a count that wrapped around, which is read as -500."""
    wrapped_count = ctypes.c_size_t(-500).value
    monkeypatch.setattr(loader, "llama_cpp_library", lambda: LibraryWithOneDevice(wrapped_count, bytes_in_all=1000))
    assert free_memory_by_device() == {"GPU0 (test card)": -500}


def test_llama_cpp_itself_tells_the_free_memory_of_its_devices() -> None:
    """Asking the real library works and gives one whole number of bytes for each of its devices."""
    free_by_device = free_memory_by_device()
    assert len(free_by_device) == len(loader.device_descriptions())
    assert all(isinstance(free_bytes, int) for free_bytes in free_by_device.values())


def test_small_model_is_measured_on_the_devices_that_hold_its_context(small_model_file: Path) -> None:
    """The measurement gives the trained context, and on each device of the context a cost for every token."""
    memory_use = measure_memory_use(small_model_file)
    assert memory_use.trained_context == SMALL_MODEL_TRAINED_CONTEXT
    skip_without_a_gpu()
    assert memory_use.devices != ()
    for device in memory_use.devices:
        assert device.each_context_token > 0
        assert device.model_weights > 0
        assert device.free_before_the_model > device.model_weights


def test_measuring_gives_back_the_memory_it_took(small_model_file: Path) -> None:
    """After the measurement every device has the memory it had before, as the model loaded to measure is closed."""
    free_before = free_memory_by_device()
    measure_memory_use(small_model_file)
    free_after = free_memory_by_device()
    for device, free_bytes in free_before.items():
        assert abs(free_after[device] - free_bytes) < BYTES_A_MEASUREMENT_MAY_BE_OFF


def test_measurement_connects_the_servers_and_loads_the_model_split_with_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """To measure, the RPC servers are connected and the model is loaded split with them, with the small context."""
    connected: list[tuple[str, ...]] = []
    opened: list[tuple[Path, tuple[str, ...], int]] = []

    def record_and_stop(model_path: Path, rpc_servers: tuple[str, ...], context_tokens: int) -> None:
        """Stand in for the load of the model: record what was asked for and end the measurement there."""
        opened.append((model_path, rpc_servers, context_tokens))
        raise RuntimeError("the test needs nothing after the load")

    monkeypatch.setattr(loader, "connect_rpc_servers", connected.append)
    monkeypatch.setattr(loader, "open_model", record_and_stop)
    with pytest.raises(RuntimeError, match="nothing after the load"):
        measure_memory_use(Path("any.gguf"), ("10.0.0.2:50052",))
    assert connected == [("10.0.0.2:50052",)]
    assert opened == [(Path("any.gguf"), ("10.0.0.2:50052",), PROBE_CONTEXT)]


def test_measurement_that_fails_closes_the_model_it_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a context of the measurement cannot be created, the model loaded to measure does not stay in memory."""
    probe = ModelThatRecordsItsClosing()

    def fail_to_create_a_context(model: ModelThatRecordsItsClosing, context_tokens: int) -> dict[str, int]:
        """Stand in for the creation of one more context: fail as llama.cpp does when the memory is not enough."""
        raise RuntimeError(f"llama.cpp could not create a context of {context_tokens} tokens")

    monkeypatch.setattr(loader, "open_model", lambda model_path, rpc_servers, context_tokens: probe)
    monkeypatch.setattr(loader, "memory_another_context_takes", fail_to_create_a_context)
    with pytest.raises(RuntimeError, match="could not create a context"):
        measure_memory_use(Path("any.gguf"))
    assert probe.was_closed


def test_measuring_writes_nothing_on_the_terminal(small_model_file: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """Measuring silences llama.cpp itself, because it runs first, with the screen already drawn on the terminal."""
    hand_messages_back_to_the_bindings()
    measure_memory_use(small_model_file)
    written = capfd.readouterr()
    assert written.out == ""
    assert written.err == ""


def test_larger_context_takes_more_memory_than_a_smaller_one(small_model_file: Path) -> None:
    """On a device that holds the model, one more context takes memory, and more when it is larger."""
    model = load_model(small_model_file, context_tokens=PROBE_CONTEXT)
    smaller_context = memory_another_context_takes(model, SMALLER_EXTRA_CONTEXT)
    larger_context = memory_another_context_takes(model, LARGER_EXTRA_CONTEXT)
    model.close()
    devices_of_the_context = [device for device in larger_context if larger_context[device] > 0]
    skip_without_a_gpu()
    assert devices_of_the_context != []
    for device in devices_of_the_context:
        assert larger_context[device] > smaller_context[device] > 0


def test_context_llama_cpp_cannot_create_is_reported(small_model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When llama.cpp gives no context back, the measurement stops with an error that says the size it tried."""
    model = load_model(small_model_file, context_tokens=PROBE_CONTEXT)
    monkeypatch.setattr(loader.llama_cpp, "llama_init_from_model", lambda model, options: None)
    with pytest.raises(RuntimeError, match=f"context of {SMALLER_EXTRA_CONTEXT} tokens"):
        memory_another_context_takes(model, SMALLER_EXTRA_CONTEXT)
    model.close()


def test_model_is_loaded_with_the_context_asked_for(small_model_file: Path) -> None:
    """A context size given to the loader is the size of the context the model gets."""
    model = load_model(small_model_file, context_tokens=1024)
    assert model.n_ctx() == 1024
    model.close()


def test_llama_cpp_is_asked_for_the_given_context_with_flash_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader passes the context size on, and turns flash attention on so that each token takes a fixed amount."""
    options_given_to_llama_cpp: dict[str, Any] = {}
    monkeypatch.setattr(loader, "open_without_touching_the_terminal", options_given_to_llama_cpp.update)
    load_model(Path("any.gguf"), context_tokens=5120)
    assert options_given_to_llama_cpp["n_ctx"] == 5120
    assert options_given_to_llama_cpp["flash_attn"] is True


def test_load_connects_the_servers_and_then_opens_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading connects the RPC servers it is given before it opens the model, which is then split with them."""
    steps: list[tuple[str, object]] = []

    def record_the_opening(**options: Any) -> ModelThatRecordsItsClosing:
        """Stand in for the creation of the model: record the file asked for and hand over a model."""
        steps.append(("open", options["model_path"]))
        return ModelThatRecordsItsClosing()

    monkeypatch.setattr(loader, "connect_rpc_servers", lambda servers: steps.append(("connect", servers)))
    monkeypatch.setattr(loader, "open_without_touching_the_terminal", record_the_opening)
    monkeypatch.setattr(loader, "free_memory_by_device", lambda: {})
    load_model(Path("any.gguf"), ("10.0.0.2:50052",))
    assert steps == [("connect", ("10.0.0.2:50052",)), ("open", "any.gguf")]


def test_model_that_takes_more_than_a_device_has_is_closed_and_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device left over its memory by the load stops it: the model is closed and the error names the device."""
    model = ModelThatRecordsItsClosing()
    monkeypatch.setattr(loader, "free_memory_by_device", lambda: {"CPU (test)": 800, "GPU0 (test card)": -500})
    with pytest.raises(RuntimeError, match="more memory than GPU0 \\(test card\\) has"):
        confirm_devices_hold(model)
    assert model.was_closed


def test_model_the_devices_hold_is_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every device still has free memory after the load, the model is kept as it is."""
    model = ModelThatRecordsItsClosing()
    monkeypatch.setattr(loader, "free_memory_by_device", lambda: {"GPU0 (test card)": 0})
    confirm_devices_hold(model)
    assert not model.was_closed


def test_load_is_checked_against_the_memory_of_the_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a model ends by checking that the devices hold it, so a context too large fails at the load."""
    model = ModelThatRecordsItsClosing()
    monkeypatch.setattr(loader, "open_without_touching_the_terminal", lambda **options: model)
    monkeypatch.setattr(loader, "free_memory_by_device", lambda: {"GPU0 (test card)": -1})
    with pytest.raises(RuntimeError, match="its context has to be smaller"):
        load_model(Path("any.gguf"))
    assert model.was_closed
