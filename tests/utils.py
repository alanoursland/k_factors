# tests/utils.py
"""
Small, reusable helpers used across the K-Factors test suite.

Functions:
- unit(v): unit-normalize a vector (numpy or torch), preserving input type.
- abs_cos(u, v): absolute cosine similarity (returns float).
- perm_invariant_axis_match(B, G): best sum of |cos| over permutations; returns (score, perm).
- perm_invariant_accuracy(y_pred, split_index): best accuracy over label swap for 2-way synthetic splits.
- time_block(label, meta=None): context manager that prints wall-clock time with optional metadata.
- print_timing(label, seconds, **meta): convenience printer for timings (used by time_block).

Notes:
- Keep dependencies light; torch is optional.
- We normalize defensively with small eps to avoid division by zero.
"""

from __future__ import annotations

import json
import itertools
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor as TorchTensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TorchTensor = None  # type: ignore

ArrayLike = Union[np.ndarray, "TorchTensor"]


# ----------------------------
# Basic numeric util helpers
# ----------------------------
_EPS = 1e-12


def _is_torch(x: Any) -> bool:
    return (torch is not None) and isinstance(x, torch.Tensor)


def _to_numpy_1d(x: ArrayLike) -> np.ndarray:
    """Convert a vector (1D) to numpy array without altering shape semantics."""
    if _is_torch(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {x.shape}")
    return x


def _to_numpy_2d(x: ArrayLike) -> np.ndarray:
    """Convert a 2D array to numpy array."""
    if _is_torch(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


# ----------------------------
# Public helpers
# ----------------------------
def unit(v: ArrayLike) -> ArrayLike:
    """
    Return the unit-normalized vector, preserving input type (numpy or torch).

    Parameters
    ----------
    v : np.ndarray or torch.Tensor (1D)

    Returns
    -------
    same type as input
    """
    if _is_torch(v):
        denom = v.norm() + (_EPS if v.numel() > 0 else 1.0)
        return v / denom
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError(f"unit() expects 1D input, got shape {v.shape}")
    denom = np.linalg.norm(v) + (_EPS if v.size > 0 else 1.0)
    return v / denom


def abs_cos(u: ArrayLike, v: ArrayLike) -> float:
    """
    Absolute cosine similarity between two vectors.

    Accepts numpy arrays or torch tensors (1D). Returns float.
    """
    u_np = _to_numpy_1d(u)
    v_np = _to_numpy_1d(v)
    if u_np.shape != v_np.shape:
        raise ValueError(f"Shape mismatch: {u_np.shape} vs {v_np.shape}")
    denom = (np.linalg.norm(u_np) + _EPS) * (np.linalg.norm(v_np) + _EPS)
    return float(np.abs(np.dot(u_np, v_np)) / denom)


def perm_invariant_axis_match(
    B: ArrayLike, G: ArrayLike
) -> Tuple[float, Tuple[int, ...]]:
    """
    Best sum of absolute cosine similarities between rows of B and a permutation of rows of G.

    Parameters
    ----------
    B : (k, d) learned unit (or near-unit) vectors (rows)
    G : (k, d) ground-truth unit vectors (rows)

    Returns
    -------
    (best_score, best_perm)
      best_score: float sum of |cos| across matched rows
      best_perm : tuple of indices p such that G[p[i]] is matched to B[i]

    Notes
    -----
    - Robust to sign flips because we take absolute cosine.
    - If k is large, this O(k!) brute-force approach can be slow; tests here keep k small.
    """
    B_np = _to_numpy_2d(B)
    G_np = _to_numpy_2d(G)
    if B_np.shape != G_np.shape:
        raise ValueError(f"Shape mismatch: B {B_np.shape} vs G {G_np.shape}")
    k, d = B_np.shape
    if k == 0:
        return 0.0, tuple()

    # Normalize rows defensively
    Bn = B_np / (np.linalg.norm(B_np, axis=1, keepdims=True) + _EPS)
    Gn = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + _EPS)

    # Pairwise |cos| similarities (k x k)
    S = np.abs(Bn @ Gn.T)

    best_score = -np.inf
    best_perm: Tuple[int, ...] = tuple(range(k))
    for perm in itertools.permutations(range(k)):
        score = float(np.sum(S[np.arange(k), perm]))
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_score, best_perm


def perm_invariant_accuracy(y_pred: np.ndarray, split_index: int) -> float:
    """
    Best accuracy over label swaps for 2-way synthetic datasets where the
    first `split_index` points belong to class 0 and the rest to class 1.

    Parameters
    ----------
    y_pred : (n,) predicted integer labels (0/1 or any ints)
    split_index : int, number of points in the first (true) class

    Returns
    -------
    float in [0, 1]
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1D, got shape {y_pred.shape}")
    n = y_pred.size
    if not (0 <= split_index <= n):
        raise ValueError(f"split_index must be in [0, {n}], got {split_index}")

    first = y_pred[:split_index]
    second = y_pred[split_index:]

    # Map A: first→0, second→1
    acc_a = (np.sum(first == 0) + np.sum(second == 1)) / max(1, n)
    # Map B: first→1, second→0
    acc_b = (np.sum(first == 1) + np.sum(second == 0)) / max(1, n)

    return float(max(acc_a, acc_b))


@contextmanager
def time_block(label: str, meta: Dict[str, Any] | None = None):
    """
    Context manager to time a block and print a single-line summary.

    Example
    -------
    >>> with time_block("fit", {"n": 400, "d": 3, "K": 2, "R": 1}):
    ...     model.fit(X)

    Output
    ------
    [timing] fit {"n":400,"d":3,"K":2,"R":1} 0.123s
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print_timing(label, dt, **(meta or {}))


def print_timing(label: str, seconds: float, **meta: Any) -> None:
    """
    Print timing in a compact, machine-readable single line.

    Example:
    [timing] fit {"n":400,"d":3,"K":2,"R":1} 0.123s
    """
    meta_str = ""
    if meta:
        # Compact JSON to make it easy to parse if needed
        try:
            meta_str = " " + json.dumps(meta, separators=(",", ":"))
        except Exception:
            # Fallback: repr
            meta_str = " " + repr(meta)
    print(f"[timing] {label}{meta_str} {seconds:.3f}s")
