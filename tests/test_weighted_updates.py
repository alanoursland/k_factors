# tests/unit/test_weighted_updates.py
"""
U4 — Weighted update respects weights (Phase 1)

Covers:
- SubspaceRepresentation.update_from_points with per-point weights biases the
  learned direction toward the more heavily weighted subset.
- PPCARepresentation.update_from_points shows analogous behavior on W[:,0].

All tests run on CPU via the `torch_device` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.representations.subspace import SubspaceRepresentation
from kfactors.representations.ppca import PPCARepresentation
import utils  # abs_cos, unit, etc.


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


def _two_lines_X(n_per: int = 200, noise: float = 0.05, seed: int = 123) -> np.ndarray:
    """
    Build a dataset consisting of:
      - n_per points ~ N(0,1) * e_x + noise * N(0,I)
      - n_per points ~ N(0,1) * e_y + noise * N(0,I)
    """
    rng = np.random.default_rng(seed)
    t1 = rng.normal(size=(n_per, 1))
    t2 = rng.normal(size=(n_per, 1))
    c1 = np.hstack(
        [
            t1,
            noise * rng.normal(size=(n_per, 1)),
            noise * rng.normal(size=(n_per, 1)),
        ]
    )
    c2 = np.hstack(
        [
            noise * rng.normal(size=(n_per, 1)),
            t2,
            noise * rng.normal(size=(n_per, 1)),
        ]
    )
    X = np.vstack([c1, c2]).astype(np.float32)  # shape (2*n_per, 3)
    return X


def test_subspace_update_weight_biases_direction(seed_all, torch_device):
    device = torch_device
    X = _two_lines_X(n_per=200, noise=0.05, seed=7)
    n = X.shape[0]
    points = torch.tensor(X, dtype=torch.float32, device=device)

    # Indices
    n_per = n // 2
    idx_A = slice(0, n_per)      # ~ e_x
    idx_B = slice(n_per, n)      # ~ e_y

    # Case A: emphasize e_x points
    wA = 1.0
    wB = 0.05
    w = torch.full((n,), wB, dtype=torch.float32, device=device)
    w[idx_A] = wA

    rep = SubspaceRepresentation(dimension=3, subspace_dim=1, device=device)
    rep.update_from_points(points, weights=w)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    v = rep.basis[:, 0]
    score_x = utils.abs_cos(v, ex)
    score_y = utils.abs_cos(v, ey)
    assert score_x > 0.90 and score_x > score_y, f"Expected bias toward e_x; got |cos_x|={score_x:.3f}, |cos_y|={score_y:.3f}"

    # Case B: emphasize e_y points
    wA2 = 0.05
    wB2 = 1.0
    w2 = torch.full((n,), wA2, dtype=torch.float32, device=device)
    w2[idx_B] = wB2

    rep2 = SubspaceRepresentation(dimension=3, subspace_dim=1, device=device)
    rep2.update_from_points(points, weights=w2)

    v2 = rep2.basis[:, 0]
    score_x2 = utils.abs_cos(v2, ex)
    score_y2 = utils.abs_cos(v2, ey)
    assert score_y2 > 0.90 and score_y2 > score_x2, f"Expected bias toward e_y; got |cos_x|={score_x2:.3f}, |cos_y|={score_y2:.3f}"


def test_ppca_update_weight_biases_direction(seed_all, torch_device):
    device = torch_device
    X = _two_lines_X(n_per=200, noise=0.05, seed=11)
    n = X.shape[0]
    points = torch.tensor(X, dtype=torch.float32, device=device)

    n_per = n // 2
    idx_A = slice(0, n_per)      # ~ e_x
    idx_B = slice(n_per, n)      # ~ e_y

    # Case A: emphasize e_x points
    w = torch.full((n,), 0.05, dtype=torch.float32, device=device)
    w[idx_A] = 1.0

    rep = PPCARepresentation(dimension=3, latent_dim=1, device=device, init_variance=1.0)
    rep.update_from_points(points, weights=w)
    wvec = rep.W[:, 0]
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    score_x = utils.abs_cos(wvec, ex)
    score_y = utils.abs_cos(wvec, ey)
    assert score_x > 0.90 and score_x > score_y, f"PPCA should bias toward e_x; got |cos_x|={score_x:.3f}, |cos_y|={score_y:.3f}"

    # Case B: emphasize e_y points
    w2 = torch.full((n,), 0.05, dtype=torch.float32, device=device)
    w2[idx_B] = 1.0

    rep2 = PPCARepresentation(dimension=3, latent_dim=1, device=device, init_variance=1.0)
    rep2.update_from_points(points, weights=w2)
    wvec2 = rep2.W[:, 0]
    score_x2 = utils.abs_cos(wvec2, ex)
    score_y2 = utils.abs_cos(wvec2, ey)
    assert score_y2 > 0.90 and score_y2 > score_x2, f"PPCA should bias toward e_y; got |cos_x|={score_x2:.3f}, |cos_y|={score_y2:.3f}"

    # Optional sanity: variance should be positive
    assert float(rep.variance) > 0.0
    assert float(rep2.variance) > 0.0
