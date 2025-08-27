# SubspaceRepresentation — Design Guide

> Cluster model for **K-Subspaces / K-Lines** style methods. Each cluster is an **affine subspace** defined by a centroid $\mu \in \mathbb{R}^d$ and an **orthonormal basis** $V \in \mathbb{R}^{d\times r}$ (columns $v_1,\dots,v_r$).

---

## Purpose

* Provide a **geometric** (non-probabilistic) cluster representation.
* Support:

  * Orthogonal **distance-to-subspace** computations.
  * **Projection** of points onto the subspace (and coordinates in that basis).
  * **PCA-based updates** of $\mu$ and $V$ from assigned points (optionally weighted).
* Integrates with:

  * `SequentialPCAUpdater` (stagewise, residual PCA).
  * `PenalizedAssignment` (uses current-stage column for directional penalty).

---

## Construction & State

```python
SubspaceRepresentation(dimension: int, subspace_dim: int, device: torch.device)
```

* Validates `subspace_dim ≤ dimension`.
* Initializes:

  * Mean: `_mean ∈ ℝ^d` (zeros; inherited from `BaseRepresentation`).
  * Basis: `_basis ∈ ℝ^{d×r}` via QR on a random matrix if `r>0`; empty `(d×0)` otherwise.

### Properties

* `basis -> Tensor[(d, r)]` — orthonormal columns.
* `basis = value` — setter orthonormalizes via `orthogonalize_basis` (QR) and enforces shape `(d, r)`.
* Inherited:

  * `mean -> Tensor[(d,)]`
  * `dimension -> int`
  * `device -> torch.device`

---

## Distances (Orthogonal Residual)

```python
distance_to_point(points: Tensor[n, d]) -> Tensor[n]
```

Squared orthogonal distance to the affine subspace:

$$
\mathrm{dist}(x) = \left\| (x-\mu) - V V^\top (x-\mu) \right\|_2^2.
$$

* Centers points by `μ`.
* If `r=0`: distance reduces to $\|x-\mu\|^2$.
* Otherwise:

  * Coefficients $c = (x-\mu)^\top V$.
  * Projection $\hat{x} = \mu + V c$.
  * Residual $r = (x-\mu) - V c$.
  * Return `sum(r*r, dim=1)`.

**Shape/device checks** use `_check_points_shape` from `BaseRepresentation`.

---

## Projection

```python
project_points(points: Tensor[n, d]) -> (coeffs: Tensor[n, r], projections: Tensor[n, d])
```

* Coefficients: $c = (x-\mu)^\top V$.
* Projections: $\hat{x} = \mu + V c$.
* If `r=0`: returns empty `coeffs (n×0)` and `projections = μ` tiled.

---

## Parameter Update (Batch PCA)

```python
update_from_points(points: Tensor[n, d], weights: Optional[Tensor[n]] = None) -> None
```

* **Mean update**:

  * Unweighted: $\mu \leftarrow \text{mean}(X)$.
  * Weighted: normalized weights $w$, $\mu \leftarrow \sum_i w_i x_i$.
* **Centering**:

  * Unweighted: `centered = X - μ`.
  * Weighted: `centered = (X - μ) * sqrt(w)` for weighted PCA.
* **Basis update** (if `r>0`):

  * Computes SVD of `centered`:

    * Uses economy/full SVD heuristics depending on `n` vs `d`.
    * `V = right-singular-vectors`.
  * Sets `_basis = V[:, :r]`.
  * On SVD failure: warns and **keeps previous basis**.

> Note: This batch update is distinct from the **sequential** updater; both are valid pathways in the library.

---

## Serialization & Introspection

* `get_parameters()` → `{ 'mean': (d,), 'basis': (d, r) }` (cloned).
* `set_parameters(params)` → accepts any subset; `basis` re-orthonormalized on set.
* `__repr__` → concise summary with `dimension` and `subspace_dim`.

---

## Numerical Considerations

* **Orthonormality**: enforced on every `basis` set via QR; prevents drift and improves stability of distances and projections.
* **SVD choices**: selects economy SVD when `n < d` and `r < n` to reduce cost; falls back to full SVD otherwise.
* **Degeneracy**: if SVD fails (rare with Torch), a warning is issued and the previous basis is retained to avoid catastrophic updates.
* **r = 0**: model degenerates to centroid-only; distances are Euclidean-to-mean, projections equal `μ`.

---

## Complexity (per update)

Let $n$ be assigned points, $d$ ambient dim, $r$ subspace dim.

* Mean/centering: $O(nd)$.
* SVD:

  * Economy: $O(nd\min(n,d))$.
  * Full: $O(\min(n^2d, nd^2))$.
* Distance/projection queries: $O(ndr)$ for the matmuls.

---

## Interactions in K-Factors

* **Assignments**: `PenalizedAssignment` can compute stage-aware residuals using **partial** columns (handled there) and uses the **current-stage column** for the directional penalty signal.
* **Updates**: `SequentialPCAUpdater` updates `basis[:, stage]` from residuals and re-orthogonalizes `basis[:, :stage+1]`; this class’s batch update provides an alternative (non-staged) route.
* **State snapshots**: `ClusterState.bases` stores bases across clusters as `(K, R, d)` (note the transposition relative to `(d, r)` here).

---

## Invariants & Expectations

* `basis` columns are **unit-norm** and mutually orthogonal.
* `distance_to_point` and `project_points` assume orthonormal `basis`.
* `get_parameters()/set_parameters()` must round-trip the representation’s state.
* All tensors live on `.device`; `BaseRepresentation.to(device)` handles deep moves by parameter dict.

---

*SubspaceRepresentation offers a clean, stable geometry for affine subspace clustering: fast residual distances, robust PCA updates, and orthonormal bases that integrate smoothly with K-Factors’ staged learning and penalized assignments.*
