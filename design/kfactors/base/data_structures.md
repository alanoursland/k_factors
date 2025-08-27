# Core Data Structures — Design Guide

> This document explains the design and contracts of `data_structures.py` for the K-Factors library. It covers `ClusterState`, `AssignmentMatrix`, `DirectionTracker`, and `AlgorithmState`: what each stores, how they interact, expected shapes/dtypes, and key invariants.

---

## ClusterState

**Purpose.** A compact snapshot of all **cluster parameters** at a given iteration, stored in batched tensors so downstream code can operate vectorized.

### Fields

* `means: Tensor[(K, d)]` — cluster centers (always required).
* `n_clusters: int` — number of clusters `K`.
* `dimension: int` — ambient dimension `d`.
* Optional, representation-dependent:

  * `covariances: Tensor[(K, d, d)]` — full covariances.
  * `precisions: Tensor[(K, d, d)]` — inverse covariances.
  * `bases: Tensor[(K, R, d)]` — per-cluster orthonormal bases (e.g., subspaces).
  * `variances: Tensor[(K,)]` or `[(K, d)]` — scalar/diagonal variances.
  * `mixing_weights: Tensor[(K,)]` — mixture weights; defaults to uniform.
* `metadata: Dict[str, Any]` — free-form auxiliary info.

### Behavior & Invariants

* `__post_init__` validates `means.shape == (K, d)`.
* If `mixing_weights` is `None`, it is set to **uniform** (`1/K` each).
* `.device` returns `means.device`.
* `.to(device)` returns a **new** `ClusterState` with tensors moved to `device`.
* `update_means(new_means)` updates means **in place**.
* `get_subspace_basis(cluster_idx, dim_idx=None)`:

  * Returns `(R, d)` basis for cluster `cluster_idx`, or a single `(d,)` basis vector if `dim_idx` is provided.
  * Raises if `bases` is `None`.

**Notes.**

* `bases` layout is `(K, R, d)` (cluster-major, stage/latent index second). Many projection routines expect `(d, R)`; callers should `.transpose(-1, -2)` as needed.

---

## AssignmentMatrix

**Purpose.** A unified container for **hard** and **soft** assignments with efficient conversions and cluster-wise accessors.

### Construction

```python
AssignmentMatrix(assignments: Tensor, n_clusters: int, is_soft: bool = False)
```

* **Hard**: `assignments.shape == (n,)`, dtype integral (cast to `Long`).
* **Soft**: `assignments.shape == (n, K)`, rows must sum to **exactly** 1.0.

### Stored State

* `_hard_assignments: Optional[Tensor[(n,)]]`
* `_soft_assignments: Optional[Tensor[(n, K)]]`
* `n_clusters: int`
* `is_soft: bool`

### API

* `n_points -> int`
* `get_hard() -> Tensor[(n,)]`

  * Returns hard indices; if soft internally, uses `argmax` per row.
* `get_soft() -> Tensor[(n, K)]`

  * Returns soft weights; if hard internally, returns one-hot rows.
* `get_cluster_indices(k: int) -> Tensor[(m,)]`

  * Hard: indices where label == `k`.
  * Soft: indices where responsibility for `k` > 0.5 (heuristic threshold).
* `get_cluster_weights(k: int) -> Tensor`

  * Hard: a vector of ones for the selected indices.
  * Soft: column `k` of responsibilities (shape `(n,)`).
* `count_per_cluster() -> Tensor[(K,)]`

  * Hard: bincount (float).
  * Soft: column sums.
* `to(device) -> AssignmentMatrix`

  * Returns a **new** matrix with internal tensor moved.

### Invariants & Conventions

* **Soft normalization:** constructor asserts row sums equal 1 (no tolerance). Upstream code should ensure normalization to avoid assertion failures.
* **Thresholding in soft → indices:** the `0.5` cutoff is a pragmatic default for update steps that expect a subset; alternative policies can wrap this class or operate on `get_soft()` directly.

---

## DirectionTracker

**Purpose.** Records, per point, the sequence of **claimed directions** across K-Factors **stages** and computes **penalties** against reusing similar directions during assignment.

### State

* `n_points: int`, `n_clusters: int`, `device: torch.device`
* `claimed_directions: List[List[Tensor[(d,)]]]`

  * For each point `i`, a (possibly empty) list of **unit** direction vectors previously claimed by that point in earlier stages.
* `current_stage: int` — stage counter (used by callers when batching claims).

### API

* `add_claimed_direction(point_idx, cluster_idx, direction)`

  * Appends `direction.detach()` to `claimed_directions[point_idx]`.
* `add_claimed_directions_batch(assignments, cluster_bases)`

  * `assignments: Tensor[(n,)]` (hard).
  * `cluster_bases`: either `(K, R, d)` (take `[:, current_stage, :]`) **or** `(K, d)` (single direction per cluster).
  * For each point `i`, looks up its assigned cluster `c = assignments[i]` and appends the corresponding cluster direction for the **current stage**.
* `compute_penalty(point_idx, test_direction, penalty_type='product') -> float`

  * If point `i` has no history, returns `1.0` (no penalty).
  * Else computes absolute cosine similarities between `test_direction` and each `claimed_dir` in the point’s history:

    * `similarities = |<test, claimed>| ∈ [0,1]` (assumes unit vectors).
    * **product**: `∏ (1 - similarity)`
    * **sum**: `mean(1 - similarity)`
  * Returns a scalar in **\[0, 1]** — **smaller** means **more reuse** (stronger penalty).
* `compute_penalty_batch(test_directions, penalty_type='product') -> Tensor[(n, K)]`

  * `test_directions.shape == (n, K, d)`; row `i`, column `k` is the candidate direction (e.g., the **current** basis column for cluster `k`).
  * For each point `i` with history:

    * `similarities = |claimed_stack @ test_direction[i, k]|`
    * Aggregate by `product` or `mean(1 - sim)` as above.
  * Points with no history get **1** everywhere (neutral).
* `advance_stage()` — increments `current_stage` (not used by penalties directly but useful for callers’ bookkeeping).
* `reset()` — clears history and sets `current_stage = 0`.
* `to(device) -> DirectionTracker` — deep-copies and moves stored tensors.

### Penalty Semantics (critical)

* The tracker returns **penalty factors in \[0, 1]** where:

  * **1.0** ⇒ **no** prior alignment (no penalty),
  * **0.0** ⇒ **perfect** alignment with at least one claimed direction (max penalty).
* With `product`, penalties can decay **multiplicatively** with multiple overlaps, emphasizing **any** strong reuse.
* With `sum`, penalties reflect the **average** dissimilarity, emphasizing the **overall** distinctness.

**Usage in K-Factors.** In `PenalizedAssignment`, penalties are mixed as:

```
scale = (1 - penalty_weight) + penalty_weight * penalty  # in [1-α, 1]
penalized_distance = distance * scale
```

Thus, **lower** tracker penalty ⇒ **smaller scale** ⇒ **lower** penalized distance. If your intended semantics are “reuse should increase cost,” invert the mapping in `PenalizedAssignment` (or here) accordingly; the tracker itself is purely geometric and returns a neutral 1 for “no history”.

---

## AlgorithmState

**Purpose.** A single-iteration checkpoint bundling everything needed for diagnostics, convergence checks, and (optionally) resumable training.

### Fields

* `iteration: int`
* `cluster_state: ClusterState`
* `assignments: AssignmentMatrix`
* `objective_value: float`
* Optional:

  * `direction_tracker: Optional[DirectionTracker]`
  * `converged: bool`
  * `metadata: Dict[str, Any]`

### API

* `to(device) -> AlgorithmState` — deep move of constituent tensors/structures; preserves Python metadata by shallow copy.

---

## Interaction Patterns

* **K-Factors stage end:** after an inner loop converges for stage `t`, the algorithm aggregates **current** per-cluster directions and calls:

  ```
  direction_tracker.add_claimed_directions_batch(assignments, cluster_bases=W_or_basis)
  ```

  This “claims” the `t`-th direction for each point within its assigned cluster.
* **Next stage assignment:** `PenalizedAssignment` constructs `(n, K, d)` `current_directions` (broadcast column `t` for each cluster) and calls:

  ```
  penalties = direction_tracker.compute_penalty_batch(current_directions, penalty_type)
  ```

  to discourage repeating prior directions.

---

## Shapes & Devices (Quick Reference)

* `K`: number of clusters, `R`: latent rank / number of stages, `d`: ambient dim, `n`: #points.
* `ClusterState.means`: `(K, d)` • `.bases`: `(K, R, d)`.
* `AssignmentMatrix`:

  * Hard: `(n,)` (Long), Soft: `(n, K)` (Float, rows sum to 1).
* `DirectionTracker.claimed_directions[i][t]`: `(d,)` (unit expected).
* `compute_penalty_batch` input: `(n, K, d)` ⇒ output `(n, K)`.
* All `.to(device)` keep structures immutable-by-construction (return new objects) except `DirectionTracker.to`, which returns a **new** tracker instance with copied lists.

---

## Complexity Notes

* `AssignmentMatrix` operations are vectorized; `count_per_cluster` is O(n).
* `DirectionTracker.compute_penalty_batch`:

  * For each point with history size `H_i`, computing similarities is O(`H_i * K * d`) (batched matvecs). History typically grows with stages `≤ R`.

---

## Gotchas & Conventions

* **Unit directions assumption:** Penalty geometry uses absolute dot products as cosines; ensure basis columns and stored directions are **normalized**.
* **Soft assignment construction:** exact row-sum assertion; use normalization (and consider numerical tolerance upstream).
* **Soft → indices threshold:** `0.5` cutoff is a heuristic; algorithms relying on strict soft updates should use `get_soft()` directly and pass weights to updaters.

---

*This guide captures how the core data structures cooperate to support K-Factors’ staged learning and directional penalties, while keeping storage and operations batched and device-aware.*
