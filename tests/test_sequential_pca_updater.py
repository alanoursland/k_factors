# tests/unit/test_sequential_pca_updater.py
"""
U3 — SequentialPCAUpdater extracts top residual component (Phase 1)

Covers:
- Stage 0 update for SubspaceRepresentation: recovers the dominant axis (~e_x)
- Stage 0 update for PPCARepresentation: recovers the dominant axis (~e_x)

All tests run on CPU via the `torch_device` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.updates.sequential_pca import SequentialPCAUpdater
from kfactors.representations.subspace import SubspaceRepresentation
from kfactors.representations.ppca import PPCARepresentation
import utils  # abs_cos, unit, etc.


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


def _make_line_3d_along_x(n: int = 400, noise: float = 0.05, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n, 1))  # dominant along x
    X = np.hstack(
        [
            t,  # x component
            noise * rng.normal(size=(n, 1)),
            noise * rng.normal(size=(n, 1)),
        ]
    ).astype(np.float32)
    return X


def test_stage0_extracts_top_axis_subspace(seed_all, torch_device):
    device = torch_device
    X = _make_line_3d_along_x(n=400, noise=0.05, seed=7)
    points = torch.tensor(X, dtype=torch.float32, device=device)

    rep = SubspaceRepresentation(dimension=3, subspace_dim=1, device=device)
    updater = SequentialPCAUpdater()

    # Stage 0 should extract the dominant component (~ e_x)
    updater.update(rep, points, current_stage=0)

    # Compare learned direction to e_x
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    learned = rep.basis[:, 0]
    score = utils.abs_cos(learned, ex)
    assert score > 0.95, f"Expected alignment with e_x, got |cos|={score:.4f}"


def test_stage0_extracts_top_axis_ppca(seed_all, torch_device):
    device = torch_device
    X = _make_line_3d_along_x(n=400, noise=0.05, seed=11)
    points = torch.tensor(X, dtype=torch.float32, device=device)

    rep = PPCARepresentation(dimension=3, latent_dim=1, device=device, init_variance=1.0)
    updater = SequentialPCAUpdater()

    updater.update(rep, points, current_stage=0)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    w = rep.W[:, 0]
    score = utils.abs_cos(w, ex)
    assert score > 0.95, f"Expected alignment with e_x, got |cos|={score:.4f}"
