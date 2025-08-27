# tests/test_scaffold_smoke.py
"""
Phase 0 scaffold smoke tests.

Goal: ensure helpers import, timing prints, permutation utilities behave,
and deterministic seeding patterns are in place. No algorithm logic yet.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np
import pytest


def test_utils_imports(seed_all):
    # Import helper modules
    import utils
    import data_gen

    # Check expected helper names exist
    for name in [
        "unit",
        "abs_cos",
        "perm_invariant_axis_match",
        "perm_invariant_accuracy",
        "time_block",
        "print_timing",
    ]:
        assert hasattr(utils, name), f"utils.{name} should exist"

    # Data generators: functions exist (even if NotImplemented yet)
    for name in [
        "make_two_lines_3d",
        "make_aniso_gaussians",
        "make_two_planes_3d",
    ]:
        assert hasattr(data_gen, name), f"data_gen.{name} should exist"


def test_time_block_prints_duration(capsys):
    from utils import time_block

    # Do something tiny but not zero-length to ensure a measurable time is printed
    with time_block("noop", {"phase": 0}):
        time.sleep(0.01)

    captured = capsys.readouterr().out.strip()
    assert "[timing] noop" in captured
    # Look for a floating-point seconds value ending with 's'
    assert re.search(r"\s\d+\.\d{3}s$", captured) is not None, f"unexpected timing line: {captured}"


def test_seed_consistency_rng(seed_all):
    """
    Determinism sanity: two independent Generators with the same seed
    should produce identical draws. This validates our seed resolution logic.
    """
    seed = int(os.getenv("TEST_RANDOM_SEED", "1337"))
    g1 = np.random.default_rng(seed)
    g2 = np.random.default_rng(seed)

    a1 = g1.standard_normal(8)
    a2 = g2.standard_normal(8)

    assert np.array_equal(a1, a2), "Generators with same seed should match exactly"


def test_perm_invariant_axis_match_tiny():
    from utils import perm_invariant_axis_match

    # Learned axes B: e_x, e_y
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float32)
    # Ground truth G: swapped order e_y, e_x
    G = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0]], dtype=np.float32)

    score, perm = perm_invariant_axis_match(B, G)
    assert np.isclose(score, 2.0, atol=1e-7)
    assert perm == (1, 0)


def test_perm_invariant_accuracy_tiny():
    from utils import perm_invariant_accuracy

    # Perfect separation
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    acc = perm_invariant_accuracy(y_pred, split_index=3)
    assert np.isclose(acc, 1.0)

    # Mixed predictions: best swap yields 4/6 correct = 2/3
    y_pred2 = np.array([1, 1, 0, 0, 0, 1])
    acc2 = perm_invariant_accuracy(y_pred2, split_index=3)
    assert np.isclose(acc2, 2.0 / 3.0)
