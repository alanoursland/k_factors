import numpy as np
import pytest
import torch

from utils import (
    time_block,
    perm_invariant_accuracy,
)
from data_gen import make_two_planes_3d

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None


def _learned_basis_matrix(model, expected_dim: int, R: int) -> np.ndarray:
    """
    Return learned bases as an array of shape (K, R, d), unit-normalized.
    Accepts model.cluster_bases_ in either (K, d, R) or (K, R, d).
    """
    CB = model.cluster_bases_
    assert CB.ndim == 3, f"expected 3D bases tensor, got {CB.shape}"
    K = CB.shape[0]

    _, a, b = CB.shape
    rows = []
    if a == expected_dim:  # (K, d, R)
        for k in range(K):
            Vr = []
            for j in range(R):
                v = CB[k, :, j].detach().cpu().numpy()
                v = v / (np.linalg.norm(v) + 1e-12)
                Vr.append(v)
            rows.append(np.stack(Vr, axis=0))  # (R, d)
    elif b == expected_dim:  # (K, R, d)
        for k in range(K):
            Vr = []
            for j in range(R):
                v = CB[k, j, :].detach().cpu().numpy()
                v = v / (np.linalg.norm(v) + 1e-12)
                Vr.append(v)
            rows.append(np.stack(Vr, axis=0))  # (R, d)
    else:
        raise AssertionError(f"unexpected cluster_bases_ shape {CB.shape} with d={expected_dim}")

    return np.stack(rows, axis=0)  # (K, R, d)


def _principal_svs_2d(G_rows: np.ndarray, L_rows: np.ndarray) -> tuple[float, float]:
    """
    Principal angles between two 2D subspaces in R^3 via singular values.
    Inputs:
        G_rows: (2, d) ground-truth plane basis (rows are unit vectors)
        L_rows: (2, d) learned plane basis (rows are unit vectors)
    Returns:
        (s1, s2): singular values of Qg^T Ql  (cosines of principal angles), s1 >= s2
    """
    assert G_rows.shape[0] == 2 and L_rows.shape[0] == 2
    d = G_rows.shape[1]
    assert L_rows.shape[1] == d

    # Columns as orthonormal bases (d x 2)
    Qg = G_rows.T
    Ql = L_rows.T

    # Re-orthonormalize (safety)
    Qg, _ = np.linalg.qr(Qg)
    Ql, _ = np.linalg.qr(Ql)

    M = Qg.T @ Ql  # (2, 2)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.sort(s)[::-1]  # descending
    return float(s[0]), float(s[1])


@pytest.mark.skipif(KFactors is None, reason="KFactors not available")
@pytest.mark.parametrize("seed", [5, 17, 101])
def test_i3_subspace_alignment_and_orthogonality(seed_all, torch_device, seed):
    """
    I3 — Two planes in R^3 (K=2, R=2).
    - Per-cluster learned 2D subspace aligns with ground-truth plane:
      both principal-angle cosines >= 0.90 (under best cluster permutation).
    - Learned in-plane axes are near-orthogonal within each cluster: |dot| <= 0.15.
    - Permutation-invariant accuracy >= 0.85.
    - Time is printed via time_block.
    """
    n_per = 400
    d = 3
    K = 2
    R = 2
    noise = 0.05

    X, y_true, (G1, G2) = make_two_planes_3d(n_per=n_per, noise=noise, seed=seed)
    # Test-only mean separation to avoid ambiguous overlap of two planes through the origin.
    # Shift cluster 0 along +z (normal to plane 1), cluster 1 along -z.
    sep = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    X[:n_per]  += sep
    X[n_per:]  -= sep

    G_stack = np.stack([G1, G2], axis=0)  # (K, 2, d), rows are basis vectors

    model = KFactors(n_clusters=K, n_components=R, random_state=seed, verbose=0, device=torch_device)
    with time_block("I3", meta={"n": 2 * n_per, "d": d, "K": K, "R": R}):
        model.fit(X)

    # Learned bases: (K, R, d)
    L = _learned_basis_matrix(model, expected_dim=d, R=R)

    # Evaluate both cluster permutations; choose the better one
    perms = [(0, 1), (1, 0)]
    best_avg_min_sv = -np.inf
    best_perm = None
    best_pair_svs = None

    for perm in perms:
        svals = []
        for k in range(K):
            G_rows = G_stack[perm[k]]  # (2, d)
            L_rows = L[k]              # (2, d)
            s1, s2 = _principal_svs_2d(G_rows, L_rows)
            svals.append((s1, s2))
        # average of the smaller singular value across clusters (stricter)
        avg_min_sv = np.mean([min(s1, s2) for (s1, s2) in svals])
        if avg_min_sv > best_avg_min_sv:
            best_avg_min_sv = avg_min_sv
            best_perm = perm
            best_pair_svs = svals

    # Assert both singular values per cluster are strong under best permutation
    assert best_perm is not None
    for (s1, s2) in best_pair_svs:
        assert s1 >= 0.90 and s2 >= 0.90, f"subspace overlap too weak: (s1,s2)=({s1:.3f},{s2:.3f}) under perm={best_perm}"

    # Within-cluster orthogonality: |<v1, v2>| <= 0.15
    for k in range(K):
        v1 = L[k, 0] / (np.linalg.norm(L[k, 0]) + 1e-12)
        v2 = L[k, 1] / (np.linalg.norm(L[k, 1]) + 1e-12)
        dot = float(abs(np.dot(v1, v2)))
        assert dot <= 0.15, f"learned in-plane axes not orthogonal enough: |dot|={dot:.3f} (cluster {k})"

    # Accuracy (perm-invariant, block split)
    y_pred = model.predict(X).detach().cpu().numpy()
    acc = perm_invariant_accuracy(y_pred, split_index=n_per)
    assert acc >= 0.85, f"Expected accuracy >= 0.85, got {acc:.3f}"
