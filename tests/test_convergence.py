# tests/unit/test_convergence.py
"""
U5 — Convergence criteria behavior (Phase 1)

Covers:
- ChangeInObjective: patience + relative tolerance handling
- ChangeInAssignments: fraction-changed threshold + patience
- ParameterChange (means): relative Frobenius change + patience

All tests run on CPU; these are pure logic checks (no heavy tensors).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.utils.convergence import (
    ChangeInObjective,
    ChangeInAssignments,
    ParameterChange,
)
from kfactors.base.data_structures import ClusterState


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


def test_change_in_objective_patience_and_thresholds(seed_all):
    """
    We use two consecutive *small* relative changes (< rel_tol) to trigger convergence
    when patience=2. Exact equality to rel_tol does NOT count as stable in impl,
    so we keep the relative deltas safely below.
    """
    crit = ChangeInObjective(rel_tol=1e-3, abs_tol=1e-12, patience=2)

    # Start at 100.0 (initializes prev ⇒ returns False)
    assert crit.check({"iteration": 0, "objective": 100.0}) is False

    # Small relative change #1: 100.0 → 99.95  (Δ=0.05, rel=0.0005 < 1e-3)
    assert crit.check({"iteration": 1, "objective": 99.95}) is False

    # Small relative change #2: 99.95 → 99.90005  (Δ≈0.04995, rel≈0.0005)
    assert crit.check({"iteration": 2, "objective": 99.90005}) is True


def test_change_in_assignments_fraction(seed_all):
    """
    min_change_fraction is the threshold *below which* we consider the system stable.
    Using patience=2, we feed two consecutive steps with 10% changes (< 20%).
    """
    crit = ChangeInAssignments(min_change_fraction=0.2, patience=2)

    a0 = torch.zeros(10, dtype=torch.long)
    a1 = a0.clone()
    a1[0] = 1  # 1/10 changed = 0.1
    a2 = a1.clone()
    a2[1] = 1  # again 1/10 changed relative to previous

    assert crit.check({"iteration": 0, "assignments": a0}) is False
    assert crit.check({"iteration": 1, "assignments": a1}) is False  # stable_count = 1
    assert crit.check({"iteration": 2, "assignments": a2}) is True   # stable_count = 2 ⇒ True


def test_parameter_change_means_relative_frobenius(seed_all):
    """
    ParameterChange monitors relative Frobenius change of the chosen parameter.
    Two consecutive small changes (< tol) with patience=2 should converge.
    """
    tol = 1e-3
    crit = ParameterChange(tol=tol, patience=2, parameter="mean")

    K, d = 2, 3
    means0 = torch.ones(K, d)
    means1 = means0 * (1.0 + 1e-4)  # rel change ~ 1e-4
    means2 = means0 * (1.0 + 2e-4)  # another small step

    cs0 = ClusterState(means=means0, n_clusters=K, dimension=d)
    cs1 = ClusterState(means=means1, n_clusters=K, dimension=d)
    cs2 = ClusterState(means=means2, n_clusters=K, dimension=d)

    assert crit.check({"iteration": 0, "cluster_state": cs0}) is False
    assert crit.check({"iteration": 1, "cluster_state": cs1}) is False  # stable_count = 1
    assert crit.check({"iteration": 2, "cluster_state": cs2}) is True   # stable_count = 2 ⇒ True
