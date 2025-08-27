# tests/data_gen.py
"""
Tiny synthetic-data generators reused across the K-Factors test suite.

Phase 0 note:
    These are *signatures and docstrings only*. Implementations are deferred to
    Phase 1. Each function currently raises NotImplementedError.

Intended usage (Phase 1+):
    >>> X, y, G = make_two_lines_3d()
    >>> X.shape, y.shape, G.shape
    ((400, 3), (400,), (2, 3))
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

NDArray = np.ndarray


def make_two_lines_3d(
    n_per: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Construct two 1D-ish clusters in R^3 aligned with the canonical axes.

    Cluster 0 lies approximately along e_x (the x-axis) with small isotropic noise.
    Cluster 1 lies approximately along e_y (the y-axis) with small isotropic noise.

    Parameters
    ----------
    n_per : int, default=200
        Number of points per cluster (total points = 2 * n_per).
    noise : float, default=0.1
        Standard deviation of added isotropic Gaussian noise.
    seed : int or None, default=None
        If provided, use as the RNG seed for reproducibility.

    Returns
    -------
    X : (2*n_per, 3) ndarray, float32
        Data matrix where the first n_per rows belong to cluster 0 (≈ x-axis)
        and the last n_per rows belong to cluster 1 (≈ y-axis).
    y : (2*n_per,) ndarray, int64
        Ground-truth labels: [0]*n_per + [1]*n_per.
    G : (2, 3) ndarray, float32
        Ground-truth unit directions for each cluster as row vectors, in the
        same ordering as labels: [[1,0,0], [0,1,0]].

    Notes
    -----
    - Implemented in Phase 1. For Phase 0 this function is a stub.
    """
    raise NotImplementedError("Phase 1 will implement make_two_lines_3d()")


def make_aniso_gaussians(
    n_per: int,
    d: int,
    covA: NDArray,
    covB: NDArray,
    rotA: Optional[NDArray] = None,
    rotB: Optional[NDArray] = None,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
    """
    Construct two anisotropic Gaussian clusters in R^d.

    Cluster A is sampled from N(0, R_A @ covA @ R_A^T), where R_A is an optional
    rotation (orthonormal) matrix. Cluster B similarly uses covB and rotB.

    Parameters
    ----------
    n_per : int
        Number of points per cluster (total points = 2 * n_per).
    d : int
        Ambient dimension.
    covA : (d, d) ndarray
        Symmetric positive semi-definite covariance for cluster A (before rotation).
    covB : (d, d) ndarray
        Symmetric positive semi-definite covariance for cluster B (before rotation).
    rotA : (d, d) ndarray or None, default=None
        Optional orthonormal rotation to apply to covA; if None, identity is used.
    rotB : (d, d) ndarray or None, default=None
        Optional orthonormal rotation to apply to covB; if None, identity is used.
    seed : int or None, default=None
        RNG seed for reproducibility.

    Returns
    -------
    X : (2*n_per, d) ndarray, float32
        Data matrix; first n_per rows from cluster A, last n_per from B.
    y : (2*n_per,) ndarray, int64
        Ground-truth labels: [0]*n_per + [1]*n_per.
    (GA, GB) : tuple of ndarrays
        Per-cluster ground-truth top principal directions (unit vectors). In Phase 2,
        this may be computed either from the true covariances (after rotation) or from
        the generated samples with labels.

    Notes
    -----
    - Implemented in Phase 2 (post Phase 1). For Phase 0 this function is a stub.
    - The outputs GA and GB should each have shape (d,) representing the top PC.
    """
    raise NotImplementedError("Phase 2 will implement make_aniso_gaussians()")


def make_two_planes_3d(
    n_per: int,
    noise: float,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
    """
    Construct two clusters, each lying near a distinct 2D plane in R^3.

    Cluster 0: points near plane with basis G1 (2 x 3), plus isotropic noise.
    Cluster 1: points near a different plane with basis G2 (2 x 3), plus noise.

    Parameters
    ----------
    n_per : int
        Number of points per cluster (total points = 2 * n_per).
    noise : float
        Standard deviation of isotropic Gaussian noise added to each point.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    X : (2*n_per, 3) ndarray, float32
        Data matrix; first n_per rows belong to cluster 0, last n_per to cluster 1.
    y : (2*n_per,) ndarray, int64
        Ground-truth labels: [0]*n_per + [1]*n_per.
    (G1, G2) : tuple of ndarrays
        Ground-truth orthonormal bases for the two planes, each of shape (2, 3).
        Row vectors represent unit directions spanning each plane.

    Notes
    -----
    - Implemented in Phase 3. For Phase 0 this function is a stub.
    """
    raise NotImplementedError("Phase 3 will implement make_two_planes_3d()")
