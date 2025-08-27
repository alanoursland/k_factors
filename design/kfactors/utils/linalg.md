Here’s a design doc for **`linalg.py`**, focusing on what utilities it provides, how they’re meant to be used inside the K-Factors library, and where to be careful:

---

# Linear Algebra Utilities (`linalg.py`)

This module implements **numerically robust linear algebra helpers** that are used across K-Factors algorithms for PCA, subspace extraction, and probabilistic model updates. It provides stable wrappers for common matrix operations and fallbacks for cases where Torch’s built-ins may fail.

---

## Purpose

* **Numerical stability**: Avoids failures from `torch.linalg.svd` and `torch.linalg.eigh` by introducing safe fallbacks.
* **Reusability**: Provides standardized, consistent routines for basis orthogonalization, matrix decompositions, and system solving.
* **Flexibility**: Exposes multiple algorithms for the same operation (QR vs Gram–Schmidt, eigendecomposition vs SVD) depending on needs.

These utilities underpin critical components of the library, especially:

* **`SequentialPCAUpdater`** → uses `safe_svd`, `orthogonalize_basis`.
* **`PPCARepresentation`** → uses `safe_svd`, `safe_eigh`.
* **Subspace representations** → rely on orthogonalization and projection utilities.

---

## Functions

### 1. Basis Management

* **`orthogonalize_basis(basis, tol=1e-7)`**
  Uses QR decomposition to orthonormalize basis vectors.

  * Detects **rank deficiency** and truncates basis accordingly (warns if fewer directions than requested).
  * Used to stabilize sequential PCA and PPCA updates.

* **`gram_schmidt(vectors, normalize=True)`**
  Classic Gram–Schmidt orthogonalization.

  * More explicit (but less numerically stable) than QR.
  * Can optionally skip normalization.

* **`project_to_orthogonal_complement(vectors, basis)`**
  Projects input vectors onto the complement of a given subspace basis.

  * Used when extracting residual components in sequential factor models.

---

### 2. Decomposition Wrappers

* **`safe_svd(matrix, full_matrices=False, max_iter=100)`**
  Robust wrapper for SVD with multiple fallbacks:

  1. Standard `torch.linalg.svd`.
  2. Retry in **double precision**.
  3. Fallback to eigendecomposition of Gram matrix (`AAᵀ` or `AᵀA`).

  * Guarantees a result, but may degrade accuracy.

* **`safe_eigh(matrix, tol=1e-10)`**
  Robust eigenvalue decomposition for symmetric matrices:

  1. Symmetrizes input (warns if not symmetric).
  2. Uses `torch.linalg.eigh`.
  3. Falls back to **power iteration** if eigendecomposition fails.

* **`power_iteration_eigh(matrix, k, max_iter=100, tol=1e-6)`**
  Computes top-`k` eigenpairs using iterative power method with deflation.

  * Used only as a last resort, but ensures progress when `eigh` is unstable.

---

### 3. Derived Operations

* **`matrix_sqrt(matrix, method='eigh')`**
  Computes square root of a PSD matrix using eigen- or SVD-based methods.

  * Negative eigenvalues are clamped to zero.
  * Useful in covariance and metric transformations.

* **`low_rank_approx(matrix, rank)`**
  Produces a rank-`r` approximation via truncated SVD.

  * Returns `U_r, diag(S_r)` so that `U_r @ diag(S_r) @ U_rᵀ` approximates input.
  * Common in dimensionality reduction or initializing subspaces.

---

### 4. Linear System Solvers

* **`solve_linear_system(A, b, method='lstsq')`**
  Unified interface for solving linear systems:

  * **`'lstsq'`** → least-squares solver (general case).
  * **`'cholesky'`** → efficient if `A` is symmetric positive definite.
  * **`'qr'`** → QR decomposition route.
  * Falls back to **Moore–Penrose pseudoinverse** if all else fails.

---

## Design Considerations

* **Fail-safe philosophy**: Every function tries multiple strategies before giving up, so algorithms can continue running even in degenerate cases.
* **Warning-based transparency**: Emits warnings on degraded numerical conditions (rank deficiency, failed SVD, non-symmetric input).
* **Performance tradeoff**: Fallbacks like `power_iteration_eigh` are slower but prevent hard crashes during clustering.

---

## Role in K-Factors

* **Sequential updates** (PCA, PPCA, subspace): rely on `safe_svd` for extracting top components and `orthogonalize_basis` for maintaining orthogonality.
* **Distance computations**: `safe_eigh` stabilizes covariance-based metrics in PPCA.
* **Regularization**: `project_to_orthogonal_complement` ensures new directions are orthogonal to previously extracted ones in sequential extraction.
* **Robustness**: These utilities make the whole library more resistant to numerical edge cases (e.g., small sample size, near-singular covariance).

---

✅ **In short**: `linalg.py` is the **numerical backbone** of K-Factors, ensuring that clustering doesn’t collapse due to linear algebra failures. It emphasizes *safety and stability* over raw performance.

