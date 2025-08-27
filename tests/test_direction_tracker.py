# tests/unit/test_direction_tracker.py
"""
U1 — DirectionTracker penalties (Phase 1)

Covers:
- Product penalty with orthogonal / parallel / diagonal directions
- Mean ("sum") penalty normalization
- Zero-weight claims are ignored
- Batch penalties match per-point scalar penalties

All tests run on CPU via the `torch_device` fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.base.data_structures import DirectionTracker
import utils  # unit(), abs_cos(), etc.


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


def _axes(device):
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)
    diag_xy = (ex + ey) / math.sqrt(2.0)
    return ex, ey, ez, diag_xy


def test_penalty_product_simple_orth_parallel(seed_all, torch_device):
    device = torch_device
    ex, ey, ez, diag_xy = _axes(device)

    # One point, three clusters (cluster ids unused by DirectionTracker's penalty)
    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    # Prior claims: e_x with weight 1, e_y with weight 1
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=1.0)

    # Product penalty:
    # - parallel to e_x: (1 - 1)*(1 - 0) = 0
    p_par = tr.compute_penalty(point_idx=0, test_direction=ex, penalty_type="product")
    # - orthogonal (e_z): (1 - 0)*(1 - 0) = 1
    p_ortho = tr.compute_penalty(point_idx=0, test_direction=ez, penalty_type="product")
    # - diagonal in xy: (1 - sqrt(1/2))^2 ≈ 0.085786...
    p_diag = tr.compute_penalty(point_idx=0, test_direction=diag_xy, penalty_type="product")

    assert np.isclose(p_par, 0.0, atol=1e-6)
    assert np.isclose(p_ortho, 1.0, atol=1e-6)
    expected_diag = (1.0 - math.sqrt(0.5)) ** 2
    assert np.isclose(p_diag, expected_diag, atol=1e-6)


def test_penalty_mean_normalization(seed_all, torch_device):
    device = torch_device
    ex, ey, ez, diag_xy = _axes(device)

    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=1.0)

    # Mean ("sum") penalty:
    # p = 1 - (sum_i w_i |cos|) / (sum_i w_i)
    p_par = tr.compute_penalty(point_idx=0, test_direction=ex, penalty_type="sum")  # 1 - (1+0)/2 = 0.5
    p_ortho = tr.compute_penalty(point_idx=0, test_direction=ez, penalty_type="sum")  # 1 - 0 = 1
    p_diag = tr.compute_penalty(point_idx=0, test_direction=diag_xy, penalty_type="sum")  # 1 - (√½ + √½)/2

    assert np.isclose(p_par, 0.5, atol=1e-6)
    assert np.isclose(p_ortho, 1.0, atol=1e-6)
    expected_diag = 1.0 - math.sqrt(0.5)
    assert np.isclose(p_diag, expected_diag, atol=1e-6)


def test_zero_weight_claim_ignored(seed_all, torch_device):
    device = torch_device
    ex, ey, ez, diag_xy = _axes(device)

    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    # Prior claim with zero weight should be ignored completely
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=0.0)

    for cand in (ex, ey, ez, diag_xy):
        p_prod = tr.compute_penalty(point_idx=0, test_direction=cand, penalty_type="product")
        p_mean = tr.compute_penalty(point_idx=0, test_direction=cand, penalty_type="sum")
        assert np.isclose(p_prod, 1.0, atol=1e-12)
        assert np.isclose(p_mean, 1.0, atol=1e-12)


@pytest.mark.parametrize("ptype", ["product", "sum"])
def test_batch_matches_singleton(seed_all, torch_device, ptype):
    device = torch_device
    ex, ey, ez, diag_xy = _axes(device)

    tr = DirectionTracker(n_points=1, n_clusters=3, device=device)
    # History: e_x with weight 1, e_y with weight 0.5
    tr.add_claimed_direction(point_idx=0, cluster_idx=0, direction=ex, weight=1.0)
    tr.add_claimed_direction(point_idx=0, cluster_idx=1, direction=ey, weight=0.5)

    # Build (n=1, K=3, d=3) candidate tensor
    candidates = torch.stack([ex, ey, diag_xy], dim=0).unsqueeze(0)  # (1,3,3)

    # Batch penalties
    P = tr.compute_penalty_batch(candidates, penalty_type=ptype)  # (1,3)
    p_batch = P[0].detach().cpu().numpy()

    # Scalar penalties
    p_scalar = np.array([
        tr.compute_penalty(point_idx=0, test_direction=ex, penalty_type=ptype),
        tr.compute_penalty(point_idx=0, test_direction=ey, penalty_type=ptype),
        tr.compute_penalty(point_idx=0, test_direction=diag_xy, penalty_type=ptype),
    ])

    assert np.allclose(p_batch, p_scalar, atol=1e-7), f"batch {p_batch} vs scalar {p_scalar}"
