"""Shared test setup: provides the small model the tests load, and runs the tests written as coroutines."""

import asyncio
import hashlib
import inspect
from pathlib import Path
from urllib.request import urlopen

import pytest

SMALL_MODEL_URL = (
    "https://huggingface.co/ggml-org/test-model-stories260K/resolve/"
    "479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf"
)
SMALL_MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
SMALL_MODEL_FILE = Path(__file__).parent.parent / "storage" / "models" / "stories260K-f32.gguf"
DOWNLOAD_TIMEOUT_SECONDS = 30


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run a test written as a coroutine to its end; pytest alone would only create it, and nothing would run."""
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    fixture_names = inspect.signature(pyfuncitem.obj).parameters
    fixtures = {name: pyfuncitem.funcargs[name] for name in fixture_names}
    asyncio.run(pyfuncitem.obj(**fixtures))
    return True


def sha256_of(content: bytes) -> str:
    """Return the SHA-256 of the content, as hexadecimal text."""
    return hashlib.sha256(content).hexdigest()


def small_model_is_ready() -> bool:
    """Tell whether the small model is on disk and is exactly the expected file."""
    if not SMALL_MODEL_FILE.exists():
        return False
    return sha256_of(SMALL_MODEL_FILE.read_bytes()) == SMALL_MODEL_SHA256


def download_small_model() -> bytes:
    """Download the small model and return its content, failing the test when the download fails."""
    try:
        with urlopen(SMALL_MODEL_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            return response.read()
    except OSError as error:
        pytest.fail(f"could not download the small test model from {SMALL_MODEL_URL}: {error}")


def fetch_small_model() -> None:
    """Download the small model, check that it is the expected file and save it."""
    content = download_small_model()
    if sha256_of(content) != SMALL_MODEL_SHA256:
        pytest.fail(f"the file downloaded from {SMALL_MODEL_URL} is not the expected small test model")
    SMALL_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SMALL_MODEL_FILE.write_bytes(content)


@pytest.fixture(scope="session")
def small_model_file() -> Path:
    """Return the small model the tests load, downloading it first when it is missing."""
    if not small_model_is_ready():
        fetch_small_model()
    return SMALL_MODEL_FILE
