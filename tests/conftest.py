"""
Global pytest fixtures for K-Factors tests.

- Provides deterministic seeding across Python, NumPy, and PyTorch.
- Forces single-threaded torch to stabilize timings and reduce flakiness.
- Standardizes on CPU for all tests in Phase 0-1 (can be relaxed later).
"""

from __future__ import annotations

import os
import random
import time
import sys
from typing import Generator
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Add the project's src directory to the Python path so tests can import the code
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

def _get_seed() -> int:
    """Resolve the test seed from env or default."""
    env = os.getenv("TEST_RANDOM_SEED", "1337")
    try:
        return int(env)
    except ValueError:
        return 1337


@pytest.fixture(scope="session", autouse=True)
def seed_all() -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs once per session.

    Seed value comes from TEST_RANDOM_SEED (default 1337).
    """
    seed = _get_seed()

    # Python
    random.seed(seed)

    # NumPy: both legacy and Generator-based code paths become deterministic
    np.random.seed(seed)

    # PyTorch (if available)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # Seed all CUDA devices (even if we don't use them in tests)
            torch.cuda.manual_seed_all(seed)

        # Keep CUDNN behavior stable
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            # We do NOT force full deterministic algorithms to avoid surprising slowdowns.
            # If you need strict determinism for a specific test, set it locally in that test.
            # torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session", autouse=True)
def set_torch_threads() -> None:
    """
    Reduce PyTorch to a single thread for stability and consistent timing.
    """
    if torch is not None and hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(1)
        except Exception:
            pass  # best-effort only


@pytest.fixture(scope="function")
def rng(seed_all: None) -> Generator[np.random.Generator, None, None]:
    """
    Per-test NumPy Generator seeded from the session seed.

    Each test receives a fresh Generator (independent streams across tests,
    reproducible within a test).
    """
    gen = np.random.default_rng(_get_seed())
    yield gen


@pytest.fixture(scope="session")
def torch_device() -> "torch.device | None":
    """
    Standard device for tests. For now, we pin to CPU to avoid device drift.
    """
    if torch is None:
        return None
    return torch.device("cpu")
