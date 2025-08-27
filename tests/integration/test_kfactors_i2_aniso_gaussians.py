import numpy as np
import pytest
import torch

from utils import (
    time_block,
    perm_invariant_axis_match,
    perm_invariant_accuracy,
)
from data_gen import make_aniso_gaussians

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None


def _learned_basis_as_rows(model, stage: int, expected_dim: int) -> np.ndarray:
    """
    Returns a (K, d) array of unit vectors for the given stage from model.cluster_bases_,
    handling either (K, d, R) or (K, R, d) orientation internally.
    """
    CB = model.cluster_bases_
    assert CB.ndim == 3, f"expected 3D bases tensor, got {CB.shape}"
    K = CB.shape[0]

    # Accept either (K, d, R) or (K, R, d)
    _, a, b = CB.shape
    if a == expected_dim:  # (K, d, R)
        # take column `stage` from each (d, R) matrix and return as row vector
        B = []
        for k in range(K):
            v = CB[k, :, stage].detach().cpu().numpy()
            v = v / (np.linalg.norm(v) + 1e-12)
            B.append(v)
        return np.stack(B, axis=0)
    elif b == expected_dim:  # (K, R, d)
        B = []
        for k in range(K):
            v = CB[k, stage, :].detach().cpu().numpy()
            v = v / (np.linalg.norm(v) + 1e-12)
            B.append(v)
        return np.stack(B, axis=0)
    else:
        raise AssertionError(f"unexpected cluster_bases_ shape {CB.shape} with d={expected_dim}")


@pytest.mark.skipif(KFactors is None, reason="KFactors not available")
@pytest.mark.parametrize("seed", [3, 11, 47])
def test_i2_alignment_and_accuracy(seed_all, torch_device, seed):
    """
    I2 — Two anisotropic Gaussians in d=10, K=2, R=1.
    - Learned stage-0 direction per cluster aligns with that cluster's top PC (|cos| >= 0.90).
    - Permutation-invariant accuracy >= 0.85.
    - Time is printed via time_block.
    """
    d = 10
    n_per = 400
    K = 2
    R = 1

    # Strongly anisotropic, distinct top axes without extra rotations
    # Cluster A: top variance along dim 0
    # Cluster B: top variance along dim 1
    covA = np.diag([6.0, 1.0, 1.0] + [0.5] * (d - 3)).astype(float)
    covB = np.diag([1.0, 6.0, 1.0] + [0.5] * (d - 3)).astype(float)

    X, y_true, (GA, GB) = make_aniso_gaussians(
        n_per=n_per, d=d, covA=covA, covB=covB, rotA=None, rotB=None, seed=seed
    )

    # Inject a small mean separation along a dimension orthogonal to the target PCs
    # (keeps the ground-truth top directions GA/GB unchanged, but makes clusters separable)
    sep = np.zeros(d, dtype=np.float32)
    sep[-1] = 3.0
    X[:n_per] += sep
    X[n_per:] -= sep

    # Fit
    model = KFactors(n_clusters=K, n_components=R, random_state=seed, verbose=0, device=torch_device)
    with time_block("I2", meta={"n": 2 * n_per, "d": d, "K": K, "R": R}):
        model.fit(X)

    # Alignment: learned stage-0 vector per cluster vs the GT top PC per cluster
    B = _learned_basis_as_rows(model, stage=0, expected_dim=d)       # (K, d)
    G = np.stack([GA, GB], axis=0).astype(np.float32)                # (K, d)

    best_score, best_perm = perm_invariant_axis_match(B, G)
    avg_abs_cos = best_score / K
    assert avg_abs_cos >= 0.90, f"Expected avg |cos| >= 0.90, got {avg_abs_cos:.3f} (perm={best_perm})"

    # Accuracy (perm-invariant block split)
    y_pred = model.predict(X).detach().cpu().numpy()
    acc = perm_invariant_accuracy(y_pred, split_index=n_per)
    assert acc >= 0.85, f"Expected accuracy >= 0.85, got {acc:.3f}"


@pytest.mark.skipif(KFactors is None, reason="KFactors not available")
def test_i2_objective_monotone_basic(seed_all, torch_device):
    """
    Sanity: final objective should not exceed initial objective by more than a tiny epsilon.
    """
    d = 10
    n_per = 300
    K = 2
    R = 1
    seed = 7

    covA = np.diag([6.0, 1.0, 1.0] + [0.5] * (d - 3)).astype(float)
    covB = np.diag([1.0, 6.0, 1.0] + [0.5] * (d - 3)).astype(float)

    X, y_true, _ = make_aniso_gaussians(
        n_per=n_per, d=d, covA=covA, covB=covB, rotA=None, rotB=None, seed=seed
    )
    # Inject a small mean separation along a dimension orthogonal to the target PCs
    # (keeps the ground-truth top directions GA/GB unchanged, but makes clusters separable)
    sep = np.zeros(d, dtype=np.float32)
    sep[-1] = 3.0
    X[:n_per] += sep
    X[n_per:] -= sep

    model = KFactors(n_clusters=K, n_components=R, random_state=seed, verbose=0, device=torch_device)
    with time_block("I2-objective", meta={"n": 2 * n_per, "d": d, "K": K, "R": R}):
        model.fit(X)

    # Not strictly monotone by iteration necessarily, but last <= first is a mild sanity check
    if hasattr(model, "history_") and len(model.history_) >= 2:
        first = model.history_[0].objective_value
        last = model.history_[-1].objective_value
        assert last <= first + 1e-6, f"final objective {last:.6f} > initial {first:.6f}"
