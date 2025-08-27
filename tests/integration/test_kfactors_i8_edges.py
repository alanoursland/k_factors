import numpy as np
import pytest
import torch

from utils import time_block
from data_gen import make_two_lines_3d

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None

pytestmark = pytest.mark.skipif(KFactors is None, reason="KFactors not importable")

def _finite_inertia_or_objective(model, X) -> bool:
    """
    Some algorithms (e.g., KFactors sequential) don't fill history_ → inertia_ fails.
    Fall back to computing objective via current representations and predicted assignments.
    """
    try:
        return np.isfinite(model.inertia_)
    except Exception:
        import torch
        Xt = torch.as_tensor(X, dtype=torch.float32, device=model.device)
        a = model.predict(Xt)
        val = model.objective.compute(Xt, model.representations, a).item()
        return np.isfinite(val)

def _resolved_seed(val, default=1337) -> int:
    return int(val) if isinstance(val, (int, np.integer)) else int(default)


def _finite(x) -> bool:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return np.isfinite(arr).all()


def _basis_shape_and_norms_ok(cluster_bases, K: int, d: int, R: int) -> None:
    """
    cluster_bases expected shape: (K, d, R).
    Each column should have non-trivial norm (> 1e-9) and be finite.
    """
    assert isinstance(cluster_bases, torch.Tensor)
    assert cluster_bases.ndim == 3, f"Expected 3D tensor, got {cluster_bases.shape}"
    assert cluster_bases.shape == (K, d, R), f"Expected {(K, d, R)}, got {cluster_bases.shape}"
    B = cluster_bases.detach().cpu().numpy()
    assert np.isfinite(B).all(), "Non-finite values in cluster_bases_"
    for k in range(K):
        for r in range(R):
            col = B[k, :, r]
            nrm = np.linalg.norm(col)
            assert nrm > 1e-9, f"Near-zero norm for basis vec (k={k}, r={r})"


def test_i8_small_n_3d_two_lines(seed_all, torch_device):
    """
    Very small n; ensure fit completes, inertia is finite, and bases have sane shape/norms.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)
    X, y, G = make_two_lines_3d(n_per=3, noise=0.05, seed=seed)  # 6 points total

    model = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device, max_iter=30)
    with time_block("I8-small-n", meta={"n": X.shape[0], "d": X.shape[1], "K": 2, "R": 1}):
        model.fit(X)

    # Finite objective and fitted flag
    assert _finite_inertia_or_objective(model, X), "Objective is not finite"
    assert getattr(model, "fitted_", False), "Model did not set fitted_ flag"
    # KFactors uses stage_history_ instead of history_
    if hasattr(model, "stage_history_"):
        assert len(model.stage_history_) >= 1, "Empty stage_history_ after fit"

    # Bases sane
    _basis_shape_and_norms_ok(model.cluster_bases_, K=2, d=X.shape[1], R=1)


def test_i8_high_dim_low_n(seed_all, torch_device):
    """
    High-d / low-n regime: n=100, d=200, K=3, R=2.
    Just ensure stability (no exception), finite objective, and sane bases.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)
    rng = np.random.default_rng(seed)

    n = 100
    d = 200
    K = 3
    R = 2

    # Simple synthetic mixture: K means, isotropic noise
    means = rng.normal(scale=2.0, size=(K, d))
    counts = [n // K] * K
    counts[0] += n - sum(counts)  # adjust remainder

    parts = []
    for k in range(K):
        Xk = means[k] + rng.normal(scale=0.3, size=(counts[k], d))
        parts.append(Xk)
    X = np.vstack(parts).astype(np.float32, copy=False)

    model = KFactors(n_clusters=K, n_components=R, random_state=seed, device=device, max_iter=40)
    with time_block("I8-hdln", meta={"n": n, "d": d, "K": K, "R": R}):
        model.fit(X)

    assert _finite_inertia_or_objective(model, X), "Objective is not finite"
    _basis_shape_and_norms_ok(model.cluster_bases_, K=K, d=d, R=R)


def test_i8_colinear_points(seed_all, torch_device):
    """
    Degenerate geometry: all points colinear in 2D (plus tiny noise).
    Over-cluster on purpose; we only assert stability/finite objective.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)
    rng = np.random.default_rng(seed)

    n = 200
    t = rng.normal(size=(n, 1))
    X = np.hstack([t, np.zeros_like(t)])  # exact line on x-axis
    X += 1e-3 * rng.normal(size=X.shape)  # tiny noise
    X = X.astype(np.float32, copy=False)

    model = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device, max_iter=30)
    with time_block("I8-colinear", meta={"n": n, "d": 2, "K": 2, "R": 1}):
        model.fit(X)

    assert _finite_inertia_or_objective(model, X), "Objective is not finite"
    _basis_shape_and_norms_ok(model.cluster_bases_, K=2, d=2, R=1)


def test_i8_zero_variance_feature(seed_all, torch_device):
    """
    Add a constant feature (zero variance). Fit should still succeed; objective finite.
    Optionally sanity check that basis weight along the constant feature is small on average.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)

    X, y, G = make_two_lines_3d(n_per=200, noise=0.05, seed=seed)  # (400, 3)
    const_col = np.ones((X.shape[0], 1), dtype=X.dtype)  # constant feature
    Xz = np.hstack([X, const_col])  # (400, 4)

    model = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device, max_iter=40)
    with time_block("I8-zero-var", meta={"n": Xz.shape[0], "d": Xz.shape[1], "K": 2, "R": 1}):
        model.fit(Xz)

    assert _finite_inertia_or_objective(model, Xz), "Objective is not finite"
    _basis_shape_and_norms_ok(model.cluster_bases_, K=2, d=Xz.shape[1], R=1)

    # Optional: check constant-dimension coefficients are small on average
    B = model.cluster_bases_.detach().cpu().numpy()  # (K, d, R)
    const_abs = np.abs(B[:, -1, 0])  # last dim coefficients, R=1
    assert np.median(const_abs) <= 1e-2, "Constant feature has unexpectedly large basis weight"
