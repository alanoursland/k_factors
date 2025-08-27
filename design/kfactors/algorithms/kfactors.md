# K-Factors Clustering — Design Notes (Draft)

> This document reverse-engineers the design from the provided source. It explains intent, moving parts, data flow, and known edge cases. Where details depend on code not shown here, I’ve added **intentionally broken links** so you can fill them in later.

---

## 1) What this module is

**Purpose.** Implements **K-Factors**, a clustering algorithm that extends K-subspaces: each cluster learns a local low-dimensional **sequential PCA basis** while a **penalty mechanism** discourages reusing the same directions for the same points across stages.

*Entry point:* the `KFactors` class.
*Objective:* `KFactorsObjective` (minimize penalized reconstruction error).

---

## 2) High-level idea

1. We want **K** clusters, each with **R = n\_components** basis vectors.
2. We learn those R vectors **sequentially by stage** (`stage = 0..R-1`).
3. At each stage:

   * Assign each point to a cluster with a **penalized distance** that depends on which directions the point has already “claimed”.
   * Update each cluster’s PPCA representation using only its currently assigned points, but **only along the current stage direction** (via a sequential PCA updater).
   * Check **convergence** using a relative change in objective.
4. When a stage converges (or hits `max_iter`), we **commit** the discovered direction: each point “claims” the direction used this stage via a `DirectionTracker`, which then influences later stages’ penalties.
5. Repeat until all R directions are extracted.

This is roughly “EM-like per stage”: **E-step:** penalized assignment; **M-step:** per-cluster sequential PCA update.

---

## 3) Public API & configuration

### Constructor

```python
KFactors(
  n_clusters: int,
  n_components: int,
  init: str | Any = 'k-means++',
  penalty_type: str = 'product',  # 'product' | 'sum'
  penalty_weight: float = 0.9,    # 0=no penalty, 1=max penalty
  max_iter: int = 30,
  tol: float = 1e-4,
  verbose: int = 0,
  random_state: Optional[int] = None,
  device: Optional[torch.device] = None
)
```

### Fit / Transform

* `fit(X)` — runs the R stages; populates `labels_`, `cluster_bases_`, `stage_history_`.
* `transform(X)` — concatenates **cluster-local coordinates** (size `n_clusters * n_components`).
* `inverse_transform(Z, assignments=None)` — reconstructs in input space using `μ + W z`.

### Learned attributes

* `labels_: (n_samples,)` — final hard assignments.
* `cluster_bases_: (K, R, D)` — all per-cluster bases `W`.
* `direction_tracker_` — tracks which directions a point has claimed.
* `n_iter_` — total inner iterations over all stages.
* *(Note: `cluster_centers_` is referenced in `inverse_transform` but never set; see “Limitations.”)*

---

## 4) Collaborators (and what they do)

> Many collaborators live in other modules; details are inferred. See broken links.

* **Assignment**

  * `PenalizedAssignment` — produces `(assignments, aux_info)` with optional precomputed `distances`. Implements **direction-reuse penalties** using `DirectionTracker` and `current_stage`.
    See: [PenalizedAssignment design](../assignments/penalized/README.md) *(broken)*

* **Representation**

  * `PPCARepresentation` — holds parameters `(mean, W, …)` and provides `distance_to_point()` and projections.
    See: [PPCA internals](../representations/ppca/INTERNALS.md) *(broken)*

* **Update**

  * `SequentialPCAUpdater` — updates only the **current stage column** of `W` (and possibly re-orthogonalizes) from assigned points.
    See: [Sequential updater math](../updates/sequential_pca/MATH.md) *(broken)*

* **Initialization**

  * `KMeansPlusPlusInit` by default; otherwise `RandomInit` or `FromPreviousInit`. Returns initial representations with `mean` set.
    See: [Initialization contract](../initialization/CONTRACT.md) *(broken)*

* **Book-keeping**

  * `DirectionTracker` — per-point ledger of “claimed” directions across stages; exposes `add_claimed_directions_batch(...)` and a `current_stage` field.
    See: [DirectionTracker details](../base/data_structures/DirectionTracker.md) *(broken)*

* **Convergence**

  * `ChangeInObjective(rel_tol=tol, patience=2)` — stage-local early-stop.
    See: [Convergence criteria](../utils/convergence/README.md) *(broken)*

* **Objective**

  * `KFactorsObjective` — sums **squared residuals** per cluster; uses `aux_info['distances']` if provided to avoid recomputation.

---

## 5) Data flow (per stage)

```
for stage in 0..R-1:
  convergence.reset()
  direction_tracker.current_stage = stage

  for iter in 0..max_iter-1:
    assignments, aux = assignment_strategy.compute_assignments(
        X, representations,
        direction_tracker=direction_tracker,
        current_stage=stage
    )

    for cluster k:
      X_k = X[assignments == k]
      if not empty:
        update_strategy.update(representations[k], X_k, current_stage=stage)

    objective = objective.compute(X, representations, assignments, aux_info=aux)

    if convergence.check({iteration, objective, assignments, cluster_state}):
      break

  direction_tracker.add_claimed_directions_batch(
      assignments,
      stack([rep.W[:, stage] for rep in representations])
  )

  stage_history_.append({ stage, iterations, objective, converged })
```

Key side-effects:

* `representations[k].W[:, stage]` is finalized at stage end.
* The **penalty state** updates via `DirectionTracker`.

---

## 6) Objective definition

For each cluster `k` with assigned set `S_k`, error is:

$$
\sum_{i \in S_k} \|x_i - \Pi_k(x_i)\|^2
$$

where $\Pi_k$ projects onto the subspace defined by `mean` and `W`. If `aux_info['distances']` exists, the module trusts it (computed during assignment) to avoid recomputing `distance_to_point`. This keeps **fit** tight and avoids duplicate passes over data.

> Missing: exact form of the **penalty** in the assignment distance (product vs sum).
> See: [Penalty math](../assignments/penalized/PENALTY.md) *(broken)*

---

## 7) Penalty mechanism (intent)

* **Goal:** Encourage **diverse subspace usage** per point across stages; a point that already claimed directions $\{w^{(0)},...,w^{(s-1)}\}$ should be discouraged from picking the same or similar direction again at stage `s`.
* **Knobs:** `penalty_type ∈ {product, sum}`, `penalty_weight ∈ [0,1]`.
* **State:** `DirectionTracker` remembers claimed directions per point and cluster.
* **Where applied:** Only in the **assignment** step — distances are warped by the penalty; the objective value itself is still **pure reconstruction error** (penalty affects optimization path, not the metric).

---

## 8) Convergence behavior

* Convergence is **per stage** using `ChangeInObjective` on a **relative tolerance** `tol` with **patience=2** (requires a couple of stable iterations).
* If a stage fails to converge in `max_iter`, a **warning** is issued but the algorithm proceeds (direction is still “claimed”).

---

## 9) Transform & inverse\_transform

* **transform(X):**

  1. Predict hard assignments with current representations.
  2. For each cluster $k$: project centered points onto `W_k` producing local coords $z \in \mathbb{R}^R$.
  3. Concatenate per-cluster blocks into a `(n_samples, K*R)` matrix; only the block for the point’s assigned cluster is non-zero.

* **inverse\_transform(Z, assignments=None):**

  * Reconstruct points by `μ_k + W_k z_k` for each assigned cluster block.
  * If `assignments` not given, it **infers** them by picking the first non-zero block per row.

> **Caveat:** The implementation allocates `reconstructed` using `self.cluster_centers_.shape[1]`, but `cluster_centers_` is never set (assignment is commented out). See “Limitations.”

---

## 10) Error handling, logging, device

* Uses `self._validate_data(X)` (from `BaseClusteringAlgorithm`) to ensure tensor type/device.
* Respects `device` for all buffers.
* `verbose` controls stage and iteration logs (`>=2` prints per-iteration objective).

---

## 11) Complexity (rough sketch)

Let $N$ points, dimension $D$, clusters $K$, components $R$, iterations $T$ per stage (avg).

* **Assignments:** $O(NKD)$ per iteration (distance eval), but often cheaper if using cached `aux_info['distances']`.
* **Updates:** For each cluster, sequential PCA update on $n_k \times D$ slice; rough $O(\sum_k n_k D)$ with low-rank updates restricted to one column per stage.
* **Total:** $O(R \cdot T \cdot NKD)$ dominant term, with a smaller update overhead.

---

## 12) Edge cases & behavior

* **Empty clusters:** If `X_k` is empty for cluster `k`, its representation is **not** updated that iteration.
* **Non-converged stages:** Allowed; direction is still claimed and we move on.
* **Penalty off:** `penalty_weight=0` should reduce to a K-subspaces-style procedure with sequential PCA (behavior depends on updater).
* **Soft vs hard assignments:** The code constructs `AssignmentMatrix(..., is_soft=False)`—this implementation is **hard-assignment only**.

---

## 13) Extensibility seams

* Swap in different objectives by replacing `self.objective`.
* Different penalties via `PenalizedAssignment` params (and potentially new types).
* Different per-stage updaters (e.g., robust PCA, incremental SVD).
* Alternative initializations (`FromPreviousInit`, spectral seeds, etc.).
* Add **soft assignments** by extending `AssignmentMatrix` and the updater to handle weights.
* Add **regularization** of `W` (sparsity, orthogonality constraints beyond re-orth) in the updater.

---

## 14) Invariants & contracts

* `representations[k]` must expose:

  * `.mean: (D,)`
  * `.W: (D, R)`
  * `.distance_to_point(points): (n_points,)`
  * `.get_parameters()['mean']` during init.
* `SequentialPCAUpdater.update(rep, X_k, current_stage)` must update only the `current_stage` column and keep `W` **orthonormal** across columns seen so far.
  See: [Updater contract](../updates/sequential_pca/CONTRACT.md) *(broken)*

---

## 15) Known limitations / TODOs

1. **`cluster_centers_` missing.**

   * `inverse_transform` uses `self.cluster_centers_.shape[1]` to size `reconstructed`, but `cluster_centers_` is never set (the assignment is commented out).
   * **Fix:** allocate via `dimension = self.representations[0].W.shape[0]` or store means in `cluster_centers_` during `fit`.
     See: [Tracking means](./design/means-and-centers.md) *(broken)*

2. **Assignment inference in `inverse_transform`.**

   * Infers cluster by “first non-zero block.” This fails if someone passes dense `Z` or if numerical zeros occur.
   * **Fix:** require `assignments` or keep them alongside `Z`.
     See: [Transform contract](./api/transform-contract.md) *(broken)*

3. **Penalty semantics underspecified here.**

   * Need formal definition for `'product'` vs `'sum'` and how `penalty_weight` modulates distance.
     See: [Penalty math](../assignments/penalized/PENALTY.md) *(broken)*

4. **Convergence on assignments.**

   * `ChangeInObjective.check` receives a `cluster_state`, but we don’t store it; also not clear how assignment oscillations are handled beyond patience=2.
     See: [Convergence details](../utils/convergence/DETAILS.md) *(broken)*

5. **Direction claiming granularity.**

   * We claim a **single direction per stage** across all clusters. If a cluster never received points in a stage, we still push a column `W[:, stage]` into the tracker; ensure it’s defined.
     See: [DirectionTracker semantics](../base/data_structures/DirectionTracker.md) *(broken)*

6. **Device & dtype hygiene.**

   * Assumes all collaborators respect `device`. Consider explicit `to(device)` moves and mixed-precision boundaries.
     See: [Device policy](./engineering/device-policy.md) *(broken)*

---

## 16) Minimal usage sketch

```python
X = torch.randn(10_000, 64, device=device)

model = KFactors(
    n_clusters=8,
    n_components=3,
    penalty_type='product',
    penalty_weight=0.9,
    max_iter=30,
    tol=1e-4,
    device=device,
    verbose=1
).fit(X)

Z = model.transform(X)
X_hat = model.inverse_transform(Z, assignments=model.labels_)
```

---

## 17) Glossary

* **Stage** — the index of the sequential component (column of `W`) currently being learned.
* **Claimed direction** — a directional vector a point used during a stage; recorded to discourage reuse later.
* **PPCA representation** — a local linear-Gaussian model parameterized by `(μ, W, σ²)`; here primarily used for subspace projection.
  See: [PPCA background](./notes/ppca-background.md) *(broken)*

---

## 18) Files you might want to add (placeholders)

* `docs/algorithms/k_factors.md` *(this file)*
* `assignments/penalized/PENALTY.md` *(broken)*
* `updates/sequential_pca/MATH.md` *(broken)*
* `base/data_structures/DirectionTracker.md` *(broken)*
* `utils/convergence/DETAILS.md` *(broken)*

---

*Last updated:* *(auto-generated draft from code comments and signatures)*
