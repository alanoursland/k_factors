# tests/integration/test_kfactors_i1_two_lines.py
"""
Phase 2 — First integration pass

I1 — Two 1D lines in 3D (happy path)
I5 — Label permutation & sign robustness
I6 — Objective trace sanity

These tests exercise an end-to-end fit on a tiny synthetic dataset, assert
directional alignment (permutation/sign invariant), clustering accuracy
(permutation invariant), and basic objective monotonicity across stages.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Project imports
try:
    from kfactors.algorithms import KFactors  # type: ignore
except Exception:  # pragma: no cover
    KFactors = None  # type: ignore

import utils  # time_block, perm_invariant_axis_match, perm_invariant_accuracy
import data_gen  # make_two_lines_3d


pytestmark = [
    pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests"),
    pytest.mark.skipif(KFactors is None, reason="KFactors not importable"),
]

def _debug_dump(model, X_np, y_pred, split_index, G, torch_device, tag="I1"):
    """
    Print useful diagnostics when accuracy dips:
      - learned vs GT bases (cosines)
      - per-cluster counts
      - block confusion wrt split_index
      - cluster means (if available)
      - a few per-point distances for misclassified examples
    """
    import numpy as np
    import torch
    from utils import abs_cos

    n = X_np.shape[0]
    K = 2

    # Bases
    B = _learned_basis_as_rows(model, stage=0, expected_dim=X_np.shape[1])
    cos00 = abs_cos(B[0], G[0])
    cos01 = abs_cos(B[0], G[1])
    cos10 = abs_cos(B[1], G[0])
    cos11 = abs_cos(B[1], G[1])
    print(f"[{tag}] Bases |cos| matrix:\n  [[{cos00:.3f}, {cos01:.3f}],\n   [{cos10:.3f}, {cos11:.3f}]]")

    # Counts
    binc = np.bincount(y_pred.astype(int), minlength=K)
    print(f"[{tag}] Pred counts per cluster: {binc.tolist()}")

    # Block confusion wrt split
    first = y_pred[:split_index]
    second = y_pred[split_index:]
    c00 = np.sum(first == 0)
    c01 = np.sum(first == 1)
    c10 = np.sum(second == 0)
    c11 = np.sum(second == 1)
    print(f"[{tag}] Confusion (rows=block[0/1], cols=pred[0/1]): [[{c00},{c01}], [{c10},{c11}]]")

    # Means if exposed
    try:
        mu = model.cluster_centers_.detach().cpu().numpy()
        print(f"[{tag}] Cluster means:\n{mu}")
    except Exception:
        print(f"[{tag}] Cluster means not available via model.cluster_centers_")

    # Per-point distances for a few misclassified points
    X_t = torch.as_tensor(X_np, device=torch_device, dtype=torch.float32)
    with torch.no_grad():
        assign, aux = model.assignment_strategy.compute_assignments(
            X_t, model.representations, current_stage=0
        )
    D = aux.get("distances", None)
    if D is None:
        print(f"[{tag}] No 'distances' in aux_info")
        return
    D = D.detach().cpu().numpy()  # shape (n, K)

    # "True" block labels for this dataset: first half=0, second=1
    y_block = np.concatenate([np.zeros(split_index, int), np.ones(n - split_index, int)])
    mism = np.where(y_pred != y_block)[0]
    if len(mism) == 0:
        print(f"[{tag}] No misclassified points under block ground truth.")
        return
    sel = mism[:10]
    print(f"[{tag}] Showing up to 10 misclassified points (idx | true_block -> pred | d0 d1):")
    for i in sel:
        print(f"  {i:4d} | {int(y_block[i])} -> {int(y_pred[i])} | {D[i,0]:.4f} {D[i,1]:.4f}")

def _learned_basis_as_rows(
        model: KFactors, 
        stage: int = 0, 
        expected_dim: int | None = None
    ) -> np.ndarray:
    """
    Extract learned 1D basis vectors for each cluster as a (K, d) array.
    Handles either (K, d, R) or (K, R, d); normalizes each row to unit length.
    If expected_dim is provided, it's used to disambiguate which axis is 'd'.
    """
    B = model.cluster_bases_
    assert B.ndim == 3, f"Expected 3D tensor for cluster_bases_, got {B.shape}"
    # Heuristic: if middle dim equals d, shape is (K, d, R); else (K, R, d)
    if expected_dim is not None:
        if B.shape[1] == expected_dim:
            # (K, d, R) → take [:, :, stage]
            mat = B[:, :, stage]
        elif B.shape[2] == expected_dim:
            # (K, R, d) → take [:, stage, :]
            mat = B[:, stage, :]
        else:
            raise ValueError(
                f"cluster_bases_ shape {tuple(B.shape)} does not match expected_dim={expected_dim}"
            )
    else:
        # Fallback heuristic (kept as a backup for future tests):
        # prefer interpreting the larger of the last two dims as 'd'
        if B.shape[1] >= B.shape[2]:
            mat = B[:, :, stage]   # treat as (K, d, R)
        else:
            mat = B[:, stage, :]   # treat as (K, R, d)
    # To numpy
    M = mat.detach().cpu().float().numpy()
    # Normalize rows
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return M / norms


def test_i1_two_lines_alignment_and_accuracy(seed_all, torch_device):
    X, y_true, G = data_gen.make_two_lines_3d(n_per=200, noise=0.05, seed=7)
    n, d = X.shape
    K, R = 2, 1

    model = KFactors(n_clusters=K, n_components=R, random_state=0, device=torch_device)

    with utils.time_block("I1-two-lines", meta={"n": n, "d": d, "K": K, "R": R}):
        model.fit(X)

    # Alignment: permutation/sign invariant
    B = _learned_basis_as_rows(model, stage=0, expected_dim=d)  # (K, d)
    best_score, best_perm = utils.perm_invariant_axis_match(B, G)  # sum of |cos|
    avg_abs_cos = best_score / K
    assert avg_abs_cos >= 0.90, f"Expected avg |cos| ≥ 0.90, got {avg_abs_cos:.3f}"

    # Clustering accuracy: permutation invariant w.r.t. labels
    # Recompute with final parameters to avoid pre-update/stale labels
    y_pred = model.predict(X).detach().cpu().numpy()
    acc = utils.perm_invariant_accuracy(y_pred, split_index=n // 2)
    if acc < 0.90:
        _debug_dump(model, X, y_pred, split_index=n // 2, G=G, torch_device=torch_device, tag="I1")
    assert acc >= 0.90, f"Expected accuracy ≥ 0.90, got {acc:.3f}"

    # Basic shape sanity (allow (K, d, R) or (K, R, d))
    CB = model.cluster_bases_
    assert CB.ndim == 3
    assert CB.shape[0] == K
    k, a, b = CB.shape
    assert {a, b} == {d, R}, f"unexpected bases shape {CB.shape}, expected the last two dims to be a permutation of (d={d}, R={R})"


def test_i5_permutation_and_sign_robustness(seed_all, torch_device):
    X, y_true, G = data_gen.make_two_lines_3d(n_per=200, noise=0.05, seed=11)
    n, d = X.shape
    K, R = 2, 1

    # Permute rows: swap the block order
    n_per = n // 2
    Xp = np.vstack([X[n_per:], X[:n_per]]).astype(np.float32)

    model = KFactors(n_clusters=K, n_components=R, random_state=0, device=torch_device)

    with utils.time_block("I5-two-lines-permuted", meta={"n": n, "d": d, "K": K, "R": R}):
        model.fit(Xp)

    # Alignment should still be strong (permutation/sign invariant against G)
    B = _learned_basis_as_rows(model, stage=0, expected_dim=d)
    best_score, _ = utils.perm_invariant_axis_match(B, G)
    avg_abs_cos = best_score / K
    assert avg_abs_cos >= 0.90, f"[perm] Expected avg |cos| ≥ 0.90, got {avg_abs_cos:.3f}"

    # Accuracy under the *current* block order (first half vs second half)
    y_pred = model.predict(Xp).detach().cpu().numpy()
    acc = utils.perm_invariant_accuracy(y_pred, split_index=n_per)
    if acc < 0.90:
        _debug_dump(model, Xp, y_pred, split_index=n // 2, G=G, torch_device=torch_device, tag="I5")
    assert acc >= 0.90, f"[perm] Expected accuracy ≥ 0.90, got {acc:.3f}"


def test_i6_objective_trace_monotone_across_stages(seed_all, torch_device):
    X, _, _ = data_gen.make_two_lines_3d(n_per=200, noise=0.05, seed=13)
    n, d = X.shape
    K, R = 2, 1

    model = KFactors(n_clusters=K, n_components=R, random_state=0, device=torch_device)

    with utils.time_block("I6-objective-trace", meta={"n": n, "d": d, "K": K, "R": R}):
        model.fit(X)

    # Expect stage_history_ to exist and contain per-stage objective summaries
    assert hasattr(model, "stage_history_"), "KFactors should record stage_history_"
    stages = model.stage_history_
    assert isinstance(stages, list) and len(stages) == R

    objs = [s["objective"] for s in stages]
    assert len(objs) == R

    # Non-increasing across stages (allow tiny epsilon)
    for t in range(1, len(objs)):
        prev, cur = objs[t - 1], objs[t]
        eps = 1e-8 * max(1.0, abs(prev))
        assert cur <= prev + eps, f"Objective increased across stages: {prev} -> {cur}"

    # Iteration count sanity
    for s in stages:
        iters = s.get("iterations", None)
        assert iters is None or (1 <= iters), f"Unexpected iterations value: {iters}"
