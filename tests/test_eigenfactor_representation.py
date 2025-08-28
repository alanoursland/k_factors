
"""
U7 — EigenFactorRepresentation (flat factors) unit tests

Covers:
- Shapes, device, and basic parameter getters/setters
- to(device) round-trip
- update_from_points with hard assignments learns per-factor directions
- per-point weights bias a factor's learned direction
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Subject under test
try:
    from kfactors.representations.eigenfactor import EigenFactorRepresentation
except Exception:  # pragma: no cover
    EigenFactorRepresentation = None  # type: ignore

# Helper utils used elsewhere in the suite (abs_cos, etc.)
import utils  # type: ignore

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


@pytest.mark.skipif(EigenFactorRepresentation is None, reason="EigenFactorRepresentation not importable")
def test_init_shapes_and_device(seed_all, torch_device):
    device = torch_device
    K, D = 5, 7
    rep = EigenFactorRepresentation(dimension=D, n_factors=K, device=device)

    assert rep.dimension == D
    assert rep.means.shape == (K, D)
    assert rep.vectors.shape == (K, D)
    assert rep.means.device.type == device.type
    assert rep.vectors.device.type == device.type

    # defaults should be finite
    assert torch.isfinite(rep.means).all()
    assert torch.isfinite(rep.vectors).all()


@pytest.mark.skipif(EigenFactorRepresentation is None, reason="EigenFactorRepresentation not importable")
def test_get_set_params_and_to_device(seed_all, torch_device):
    device = torch_device
    other = torch.device("cpu" if device.type != "cpu" else "cuda" if torch.cuda.is_available() else "cpu")

    K, D = 3, 4
    rep = EigenFactorRepresentation(dimension=D, n_factors=K, device=device)

    # Craft params
    means = torch.randn(K, D, device=device)
    vectors = torch.randn(K, D, device=device)
    rep.set_parameters({"means": means, "vectors": vectors})

    got = rep.get_parameters()
    assert torch.allclose(got["means"], means)
    assert torch.allclose(got["vectors"], vectors)

    # to(device) round-trip clones tensors on new device
    rep2 = rep.to(other)
    assert rep2.means.device.type == other.type
    assert rep2.vectors.device.type == other.type
    assert torch.allclose(rep2.means.to(device), rep.means)
    assert torch.allclose(rep2.vectors.to(device), rep.vectors)


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


@pytest.mark.skipif(EigenFactorRepresentation is None, reason="EigenFactorRepresentation not importable")
def test_update_learns_per_factor_axes(seed_all, torch_device):
    """
    Hard-assign first half (~e_x line) to factor 0 and second half (~e_y line) to factor 1.
    After update_from_points, w_0 ~ e_x and w_1 ~ e_y.
    """
    device = torch_device
    X = _two_lines_X(n_per=200, noise=0.05, seed=7)
    points = torch.tensor(X, dtype=torch.float32, device=device)
    n = points.shape[0]

    rep = EigenFactorRepresentation(dimension=3, n_factors=2, device=device)

    # Hard assignments 0/1
    assignments = torch.zeros(n, dtype=torch.long, device=device)
    assignments[n // 2 :] = 1

    # Uniform weights (implicitly inside update)
    rep.update_from_points(points, weights=None, assignments=assignments)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)

    w0 = rep.vectors[0]
    w1 = rep.vectors[1]
    score0 = utils.abs_cos(w0, ex)
    score1 = utils.abs_cos(w1, ey)

    assert score0 > 0.90, f"factor 0 should align with e_x; got |cos|={score0:.3f}"
    assert score1 > 0.90, f"factor 1 should align with e_y; got |cos|={score1:.3f}"

    # Means near 0 (construction is zero-mean)
    assert torch.norm(rep.means[0]).item() < 0.25
    assert torch.norm(rep.means[1]).item() < 0.25


@pytest.mark.skipif(EigenFactorRepresentation is None, reason="EigenFactorRepresentation not importable")
def test_weights_bias_a_single_factor(seed_all, torch_device):
    """
    Assign all points to a single factor but weight the e_x subset higher.
    The learned direction should bias toward e_x.
    """
    device = torch_device
    X = _two_lines_X(n_per=200, noise=0.05, seed=11)
    points = torch.tensor(X, dtype=torch.float32, device=device)
    n = points.shape[0]

    rep = EigenFactorRepresentation(dimension=3, n_factors=1, device=device)

    # Everyone assigned to factor 0
    assignments = torch.zeros(n, dtype=torch.long, device=device)

    # Emphasize first half (~e_x), de-emphasize second half (~e_y)
    w = torch.full((n,), 0.05, dtype=torch.float32, device=device)
    w[: n // 2] = 1.0

    rep.update_from_points(points, weights=w, assignments=assignments)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)
    v = rep.vectors[0]
    score_x = utils.abs_cos(v, ex)
    score_y = utils.abs_cos(v, ey)

    assert score_x > 0.90 and score_x > score_y, f"Expected bias toward e_x; got |cos_x|={score_x:.3f}, |cos_y|={score_y:.3f}"


@pytest.mark.skipif(EigenFactorRepresentation is None, reason="EigenFactorRepresentation not importable")
def test_update_handles_empty_factors(seed_all, torch_device):
    """
    If a factor receives no assigned points, its parameters should remain finite
    and not produce NaNs/infs.
    """
    device = torch_device
    X = _two_lines_X(n_per=50, noise=0.05, seed=5)
    points = torch.tensor(X, dtype=torch.float32, device=device)
    n = points.shape[0]

    K = 3
    rep = EigenFactorRepresentation(dimension=3, n_factors=K, device=device)

    # Assign everyone to factor 0; 1 and 2 are empty
    assignments = torch.zeros(n, dtype=torch.long, device=device)

    # Keep a copy of initial params for factors 1 and 2
    init_means = rep.means.clone()
    init_vecs = rep.vectors.clone()

    rep.update_from_points(points, weights=None, assignments=assignments)

    # Factor 0 should be finite
    assert torch.isfinite(rep.means[0]).all()
    assert torch.isfinite(rep.vectors[0]).all()

    # Unused factors: still finite; we don't assert exact equality to init since
    # implementations may apply small regularization/orth steps.
    for k in (1, 2):
        assert torch.isfinite(rep.means[k]).all()
        assert torch.isfinite(rep.vectors[k]).all()
