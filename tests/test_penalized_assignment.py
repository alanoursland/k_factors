# tests/unit/test_penalized_assignment.py
"""
U2 — PenalizedAssignment matrix (Phase 1)

We verify that:
- aux_info['penalties'] matches the DirectionTracker-weighted cosine logic
  for both 'product' and 'sum' modes.
- Removing penalized distances from aux_info is respected (keys absent).
- Claimed-direction weights affect the penalty as intended.

Setup:
- n=1 point at the origin, so residual distances are all equal (ties irrelevant).
- K=3 subspace reps in R^3 with bases:
    f0 = e_x
    f1 = e_y
    f2 = (e_x + e_y)/√2  (the 45° diagonal in the xy-plane)
- DirectionTracker history for the single point includes previously claimed
  directions and their weights; we probe penalties for all three candidates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.assignments.penalized import PenalizedAssignment
from kfactors.base.data_structures import DirectionTracker
from kfactors.representations.subspace import SubspaceRepresentation
import utils  # abs_cos, unit, etc.


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


def _make_reps(device):
    """Build three 1D subspace representations with bases ex, ey, diag_xy."""
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    diag = (ex + ey) / math.sqrt(2.0)

    reps = []
    for v in (ex, ey, diag):
        rep = SubspaceRepresentation(dimension=3, subspace_dim=1, device=device)
        rep.mean = torch.zeros(3, device=device)
        rep.basis = v.view(3, 1)  # setter will orthonormalize
        reps.append(rep)
    return reps


def test_penalties_reflected_in_aux_info_product(seed_all, torch_device):
    device = torch_device
    reps = _make_reps(device)

    # Single point at origin (residuals identical across clusters)
    X = torch.zeros(1, 3, device=device)

    # Prior claims: e_x @ weight=1.0, e_y @ weight=1.0
    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=1.0)

    assigner = PenalizedAssignment(penalty_type="product", penalty_weight=1.0)
    assignments, aux = assigner.compute_assignments(
        X, reps, direction_tracker=tr, current_stage=0
    )

    # We only care about the penalties surfaced in aux_info
    assert "penalties" in aux
    # Make sure removed keys are indeed absent
    assert "penalized_distances" not in aux
    assert "min_penalized_distances" not in aux

    p = aux["penalties"][0].detach().cpu().numpy()  # (K,)
    # Expected:
    # f0 parallel to claimed e_x → (1 - 1)*(1 - 0) = 0
    # f1 parallel to claimed e_y → (1 - 0)*(1 - 1) = 0
    # f2 at 45° to both → (1 - √½)^2
    expected_f2 = (1.0 - math.sqrt(0.5)) ** 2

    assert np.isclose(p[0].item(), 0.0, atol=1e-6)
    assert np.isclose(p[1].item(), 0.0, atol=1e-6)
    assert np.isclose(p[2].item(), expected_f2, atol=1e-6)


def test_penalties_reflected_in_aux_info_mean(seed_all, torch_device):
    device = torch_device
    reps = _make_reps(device)
    X = torch.zeros(1, 3, device=device)

    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=1.0)

    assigner = PenalizedAssignment(penalty_type="sum", penalty_weight=1.0)
    assignments, aux = assigner.compute_assignments(
        X, reps, direction_tracker=tr, current_stage=0
    )

    p = aux["penalties"][0].detach().cpu().numpy()

    # Expected (mean form): 1 - (Σ a_i |cos|) / (Σ a_i)
    # f0: 1 - (1 + 0)/2 = 0.5
    # f1: 1 - (0 + 1)/2 = 0.5
    # f2: 1 - (√½ + √½)/2 = 1 - √½
    expected_f2 = 1.0 - math.sqrt(0.5)

    assert np.isclose(p[0].item(), 0.5, atol=1e-6)
    assert np.isclose(p[1].item(), 0.5, atol=1e-6)
    assert np.isclose(p[2].item(), expected_f2, atol=1e-6)


def test_penalties_respect_claimed_weights(seed_all, torch_device):
    """
    If the second claimed direction has smaller weight (e.g., 0.5),
    the product penalty for the 45° diagonal should increase relative
    to the equal-weights case.
    """
    device = torch_device
    reps = _make_reps(device)
    X = torch.zeros(1, 3, device=device)

    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=0.5)

    assigner = PenalizedAssignment(penalty_type="product", penalty_weight=1.0)
    _, aux = assigner.compute_assignments(
        X, reps, direction_tracker=tr, current_stage=0
    )

    p = aux["penalties"][0].detach().cpu().numpy()
    # For f2:
    # (1 - 1*√½) * (1 - 0.5*√½) = (1 - 0.7071) * (1 - 0.3536) ≈ 0.292893 * 0.646447 ≈ 0.189
    expected_f2 = (1.0 - math.sqrt(0.5)) * (1.0 - 0.5 * math.sqrt(0.5))

    assert np.isclose(p[0].item(), 0.0, atol=1e-6)  # parallel to ex
    # For f1 (parallel to ey with weight 0.5): (1 - 0)*(1 - 0.5) = 0.5
    assert np.isclose(p[1].item(), 0.5, atol=1e-6)
    assert np.isclose(p[2].item(), expected_f2, atol=1e-6)
