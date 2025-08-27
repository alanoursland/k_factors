import numpy as np
import pytest
import torch

from utils import time_block, unit, perm_invariant_axis_match
from data_gen import make_two_lines_3d, make_aniso_gaussians

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None

pytestmark = pytest.mark.skipif(KFactors is None, reason="KFactors not importable")


def _resolved_seed(val, default=1337) -> int:
    """Fixture may return None; force a deterministic integer seed."""
    return int(val) if isinstance(val, (int, np.integer)) else int(default)


def _labels_equal_up_to_perm(y1: np.ndarray, y2: np.ndarray, K: int) -> bool:
    """Return True if y2 can be permuted to equal y1 exactly (K=2 path is trivial)."""
    if K == 2:
        return np.array_equal(y1, y2) or np.array_equal(y1, 1 - y2)
    import itertools
    for perm in itertools.permutations(range(K)):
        mapping = np.array(perm)
        if np.array_equal(y1, mapping[y2]):
            return True
    return False


def _bases_d_k_from_cluster_bases(cluster_bases) -> np.ndarray:
    """
    Convert model.cluster_bases_ (torch, shape (K, d, R)) to np (K, d) for R=1.
    Unit-normalize per row.
    """
    if isinstance(cluster_bases, torch.Tensor):
        B = cluster_bases.detach().cpu().numpy()
    else:
        B = np.asarray(cluster_bases)
    assert B.ndim == 3 and B.shape[2] == 1, f"Expected (K, d, 1), got {B.shape}"
    K, d, _ = B.shape
    out = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        v = B[k, :, 0]
        n = np.linalg.norm(v) + 1e-12
        out[k] = v / n
    return out


def _random_orthonormal(d: int, rng: np.random.Generator) -> np.ndarray:
    """QR-based random rotation with det=+1."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q.astype(np.float64, copy=False)


def _fixed_cov_and_rot(d: int, seed: int):
    """
    Deterministic covariances and rotations for the aniso test.
    - CovA: diag decreasing from 3.0 to 0.3
    - CovB: diag decreasing from 2.0 to 0.2
    - Rotations: random orthonormal from RNG(seed+10/11)
    """
    valsA = np.linspace(3.0, 0.3, d, dtype=np.float64)
    valsB = np.linspace(2.0, 0.2, d, dtype=np.float64)
    covA = np.diag(valsA)
    covB = np.diag(valsB)

    rngA = np.random.default_rng(seed + 10)
    rngB = np.random.default_rng(seed + 11)
    rotA = _random_orthonormal(d, rngA)
    rotB = _random_orthonormal(d, rngB)
    return covA, covB, rotA, rotB


@pytest.mark.parametrize("n_per,noise", [(300, 0.05)])
def test_i7_two_runs_same_results_on_two_lines(seed_all, torch_device, n_per, noise):
    """
    With fixed seed & same device, two runs produce same labels (up to permutation)
    and same normalized bases (up to permutation/sign) within 1e-6.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)

    X, y_true, G = make_two_lines_3d(n_per=n_per, noise=noise, seed=seed)

    model1 = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device)
    with time_block("I7-two-lines-run1", meta={"n": X.shape[0], "d": X.shape[1], "K": 2, "R": 1}):
        model1.fit(X)
    y1 = model1.predict(X).cpu().numpy()
    B1 = _bases_d_k_from_cluster_bases(model1.cluster_bases_)

    model2 = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device)
    with time_block("I7-two-lines-run2", meta={"n": X.shape[0], "d": X.shape[1], "K": 2, "R": 1}):
        model2.fit(X)
    y2 = model2.predict(X).cpu().numpy()
    B2 = _bases_d_k_from_cluster_bases(model2.cluster_bases_)

    # Labels equal up to permutation
    assert _labels_equal_up_to_perm(y1, y2, K=2), "Labels differ beyond a global permutation"

    # Bases equal up to permutation/sign: find best permutation of B2 to match B1
    score, perm = perm_invariant_axis_match(B1, B2)  # perm applies to B2 rows
    B2p = B2[list(perm)]
    # Align sign per row and compare
    for k in range(2):
        if np.dot(B1[k], B2p[k]) < 0:
            B2p[k] = -B2p[k]
        assert np.allclose(B1[k], B2p[k], atol=1e-6), f"Base vector {k} mismatch beyond 1e-6"


@pytest.mark.parametrize("n_per,d", [(300, 10)])
def test_i7_two_runs_same_results_on_aniso_gaussians(seed_all, torch_device, n_per, d):
    """
    Determinism on a slightly richer dataset (anisotropic Gaussians).
    Same seed & device => same labels up to permutation and same bases up to perm/sign.
    """
    device = torch_device
    seed = _resolved_seed(seed_all)

    covA, covB, rotA, rotB = _fixed_cov_and_rot(d, seed)

    X, y_true, (GA, GB) = make_aniso_gaussians(
        n_per=n_per, d=d, covA=covA, covB=covB, rotA=rotA, rotB=rotB, seed=seed
    )

    model1 = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device)
    with time_block("I7-aniso-run1", meta={"n": X.shape[0], "d": X.shape[1], "K": 2, "R": 1}):
        model1.fit(X)
    y1 = model1.predict(X).cpu().numpy()
    B1 = _bases_d_k_from_cluster_bases(model1.cluster_bases_)

    model2 = KFactors(n_clusters=2, n_components=1, random_state=seed, device=device)
    with time_block("I7-aniso-run2", meta={"n": X.shape[0], "d": X.shape[1], "K": 2, "R": 1}):
        model2.fit(X)
    y2 = model2.predict(X).cpu().numpy()
    B2 = _bases_d_k_from_cluster_bases(model2.cluster_bases_)

    # Labels equal up to permutation
    assert _labels_equal_up_to_perm(y1, y2, K=2), "Labels differ beyond a global permutation"

    # Bases equal up to permutation/sign
    score, perm = perm_invariant_axis_match(B1, B2)
    B2p = B2[list(perm)]
    for k in range(2):
        if np.dot(B1[k], B2p[k]) < 0:
            B2p[k] = -B2p[k]
        assert np.allclose(B1[k], B2p[k], atol=1e-6), f"Base vector {k} mismatch beyond 1e-6"
