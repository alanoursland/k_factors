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

Phase 2 note: 
    implement `make_two_lines_3d`; leave the others as stubs with clear 
    docstrings so imports succeed and later phases can fill them in.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

NDArray = np.ndarray


def make_two_lines_3d(
    n_per: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Construct two 1D line-like clusters in R^3 with small isotropic noise.

    Cluster 0 lies approximately along e_x = (1,0,0).
    Cluster 1 lies approximately along e_y = (0,1,0).
    Both are centered at the origin (means are 0) with additive Gaussian noise.

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
    """
    rng = np.random.default_rng(seed)

    # Cluster 0 ~ line along x
    t0 = rng.normal(size=(n_per, 1))
    c0 = np.hstack(
        [
            t0,  # dominant x
            noise * rng.normal(size=(n_per, 1)),
            noise * rng.normal(size=(n_per, 1)),
        ]
    )

    # Cluster 1 ~ line along y
    t1 = rng.normal(size=(n_per, 1))
    c1 = np.hstack(
        [
            noise * rng.normal(size=(n_per, 1)),
            t1,  # dominant y
            noise * rng.normal(size=(n_per, 1)),
        ]
    )

    X = np.vstack([c0, c1]).astype(np.float32)  # (2*n_per, 3)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])
    G = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    return X, y, G


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
    """
    rng = np.random.default_rng(seed)

    covA = np.asarray(covA, dtype=float)
    covB = np.asarray(covB, dtype=float)
    assert covA.shape == (d, d), f"covA must be (d,d); got {covA.shape}"
    assert covB.shape == (d, d), f"covB must be (d,d); got {covB.shape}"

    if rotA is not None:
        rotA = np.asarray(rotA, dtype=float)
        assert rotA.shape == (d, d), f"rotA must be (d,d); got {rotA.shape}"
        SigmaA = rotA @ covA @ rotA.T
    else:
        SigmaA = covA

    if rotB is not None:
        rotB = np.asarray(rotB, dtype=float)
        assert rotB.shape == (d, d), f"rotB must be (d,d); got {rotB.shape}"
        SigmaB = rotB @ covB @ rotB.T
    else:
        SigmaB = covB

    # Ground-truth top principal directions (eigenvectors of covariance)
    wA, V_A = np.linalg.eigh(SigmaA)
    wB, V_B = np.linalg.eigh(SigmaB)
    GA = V_A[:, np.argmax(wA)]
    GB = V_B[:, np.argmax(wB)]
    # Normalize to unit (just in case of numerical drift)
    GA = GA / (np.linalg.norm(GA) + 1e-12)
    GB = GB / (np.linalg.norm(GB) + 1e-12)

    mean = np.zeros(d, dtype=float)
    XA = rng.multivariate_normal(mean=mean, cov=SigmaA, size=n_per).astype(np.float32)
    XB = rng.multivariate_normal(mean=mean, cov=SigmaB, size=n_per).astype(np.float32)

    X = np.vstack([XA, XB]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    return X, y, (GA.astype(np.float32), GB.astype(np.float32))


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

    """
    rng = np.random.default_rng(seed)

    # Plane 1: span{e1, e2}
    e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    e2 = np.array([0.0, 1.0, 0.0], dtype=float)
    G1 = np.stack([e1, e2], axis=0)  # (2, 3), rows are basis vectors

    # Plane 2: create a reproducible, distinct 2D subspace and orthonormalize
    M = rng.normal(size=(3, 2))
    # Make sure it's not accidentally too close to Plane 1; if so, nudge Z axis
    if np.linalg.matrix_rank(M) < 2:
        M += np.array([[0.0, 0.0], [0.0, 0.0], [1e-1, -1e-1]])
    # Orthonormal columns via QR; then return as rows
    Q, _ = np.linalg.qr(M)  # Q: (3, 2) with orthonormal columns
    G2 = Q.T  # (2, 3), rows are orthonormal basis vectors

    # Sample coefficients in the plane and add small isotropic noise in R^3
    T1 = rng.normal(size=(n_per, 2)).astype(np.float32)
    T2 = rng.normal(size=(n_per, 2)).astype(np.float32)
    X1 = (T1 @ G1).astype(np.float32)
    X2 = (T2 @ G2).astype(np.float32)

    if noise > 0.0:
        X1 += (noise * rng.normal(size=X1.shape)).astype(np.float32)
        X2 += (noise * rng.normal(size=X2.shape)).astype(np.float32)

    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    # Ensure rows are unit-norm (QR already gives unit columns; we’re returning rows)
    G1 = (G1 / (np.linalg.norm(G1, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    G2 = (G2 / (np.linalg.norm(G2, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

    return X, y, (G1, G2)
