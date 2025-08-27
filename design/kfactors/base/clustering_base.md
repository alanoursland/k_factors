# BaseClusteringAlgorithm — Design Notes (Draft)

> This document describes the common skeleton for clustering algorithms in the K-Factors family. It’s written from the provided source; when something depends on components not shown here, you’ll see **intentionally broken links** so you can fill those in later.

---

## 1) Purpose & Scope

`BaseClusteringAlgorithm` provides the **alternating optimization** scaffold shared by all concrete algorithms:

* Creates algorithm-specific components (assignment, update, init, convergence, objective).
* Orchestrates the **assign → update → evaluate → check convergence** loop.
* Unifies device/dtype handling, random seeding, logging, and history tracking.
* Exposes a small, sklearn-style API: `fit`, `predict`, `fit_predict`, `get_params`, `set_params`.

Subclasses implement:

* `_create_components()` — instantiate strategies and objective.
* `_create_representations(data)` — construct K cluster representations after init.

See interfaces: [interfaces.py](./interfaces.md) *(broken)*

---

## 2) Public API

### Constructor

```python
BaseClusteringAlgorithm(
  n_clusters: int,
  max_iter: int = 100,
  tol: float = 1e-4,
  verbose: int = 0,
  random_state: Optional[int] = None,
  device: Optional[torch.device] = None
)
```

* **Device**: defaults to CUDA if available; otherwise CPU.
* **Seeding**: seeds both CPU and CUDA when `random_state` is provided.

### Core methods

* `fit(X, y=None) -> self` — delegates to `_fit`.
* `fit_predict(X, y=None) -> Tensor[(n,)]` — convenience wrapper.
* `predict(X) -> Tensor[(n,)]` — computes current hard assignments via `assignment_strategy` → `AssignmentMatrix` → `get_hard()`.
* `get_params() / set_params()` — sklearn compatibility.

### Properties

* `cluster_centers_ -> Tensor[(K, D)]` — current means (extracted on demand).
* `inertia_ -> float` — final objective value from `history_[-1]`.

---

## 3) Lifecycle & Data Flow

High-level lifecycle:

1. **Create components** via `_create_components()`.
2. **Initialization**:
   `self.initialization_strategy.initialize(X, K)` produces initial per-cluster seeds (format defined by the strategy) → then `_create_representations(X)` converts these seeds to concrete `ClusterRepresentation` instances.
3. **Main loop** (`iteration = 0..max_iter-1`):

   * **Assignment**: `assignment_strategy.compute_assignments(X, representations)` → returns either `assignments` or `(assignments, aux_info)`.
     Wrap into `AssignmentMatrix(assignments, K, is_soft=assignment_strategy.is_soft)`.
   * **Update**: For each cluster `k`:

     * If **soft**: fetch weights via `get_cluster_weights(k)`; threshold small weights; call `update(representation, X_k, weights, **aux_info)`.
     * If **hard**: fetch indices via `get_cluster_indices(k)`; call `update(representation, X_k, None, **aux_info)`.
   * **Objective**: `objective.compute(X, representations, assignments)` returns a scalar tensor.
   * **Book-keeping**:

     * Extract `cluster_state = _extract_cluster_state()`.
     * Append `AlgorithmState(...)` with iteration metadata to `history_`.
   * **Convergence**: `convergence_criterion.check({...})` decides whether to stop.
4. **Finalize**: set `fitted_ = True`, report timing and (if applicable) a non-convergence warning.

Related components:

* Assignment: [AssignmentStrategy](./interfaces.md#assignmentstrategy) *(broken)*
* Update: [ParameterUpdater](./interfaces.md#parameterupdater) *(broken)*
* Objective: [ClusteringObjective](./interfaces.md#clusteringobjective) *(broken)*
* Convergence: [ConvergenceCriterion](./interfaces.md#convergencecriterion) *(broken)*
* Init: [InitializationStrategy](./interfaces.md#initializationstrategy) *(broken)*

---

## 4) Representations & State

### Cluster representations

Subclasses create a list of `ClusterRepresentation` objects in `_create_representations(X)`. A representation typically exposes:

* `get_parameters()` → dict with at least `'mean'`, and optionally `'basis'`, `'covariance'`, `'variance'`.
* Any algorithm-specific fields (e.g., PPCA’s `W`, `sigma2`).

See: [ClusterRepresentation](./interfaces.md#clusterrepresentation) *(broken)*

### Extracted state (`_extract_cluster_state`)

Builds a `ClusterState` snapshot containing:

* `means: (K, D)` — always present (pulled from `rep.get_parameters()['mean']`).
* Optionally `bases`, `covariances`, `variances` — only if present in the first rep’s parameter dict; stacked across clusters when found.

Data structures reference: [data\_structures.py](./data_structures.md) *(broken)*

---

## 5) Assignments & the AssignmentMatrix

`predict(X)` and the fit loop both rely on the assignment strategy and the `AssignmentMatrix` wrapper:

* **Input flexibility**: `compute_assignments(...)` may return just `assignments` or `(assignments, aux_info)`.
* **Soft/hard**: `assignment_strategy.is_soft` toggles the update path:

  * Soft: weighted updates via `get_cluster_weights(k)`.
  * Hard: index-based updates via `get_cluster_indices(k)` and `get_hard()` for predictions.

`AssignmentMatrix` reference: [AssignmentMatrix](./data_structures.md#assignmentmatrix) *(broken)*

---

## 6) Objective & Convergence

* **Objective direction**: `objective.minimize` controls log arrow (`↓` or `↑`). The loop doesn’t negate the value; it assumes the objective knows its direction.
* **Convergence**: Delegated to `convergence_criterion.check(context)` where `context` includes:

  * `iteration`, `objective`, raw `assignments`, and `cluster_state`.
* **History**: `history_` stores `AlgorithmState` with `objective_value`, `assignments` (wrapped), and any `aux_info`.

Convergence details: [ConvergenceCriterion](./interfaces.md#convergencecriterion) *(broken)*

---

## 7) Data Validation, Device & Dtype

`_validate_data(X)` ensures:

* 2D `torch.Tensor` on `self.device`.
* Float32 dtype (casts if needed).

Device policy & mixed precision (if any): [device-policy.md](./engineering/device-policy.md) *(broken)*

---

## 8) Logging & Timing

* Per-iteration timing and objective printed when:

  * `verbose >= 2`, or
  * `verbose >= 1` and `iteration % 10 == 0`.
* On exit:

  * If not converged by `max_iter`, emits a warning.
  * Reports total fit time.

---

## 9) Extensibility Points

* Swap **assignment** (e.g., penalized, probabilistic), **update** (e.g., PPCA, robust PCA), **objective** (e.g., likelihood vs. reconstruction), **convergence** (e.g., relative change with patience), and **initialization** (e.g., k-means++, spectral).
* Implement **soft** vs **hard** variants by choosing strategies exposing `is_soft`.
* Extend `_extract_cluster_state` to capture more parameters (e.g., priors, regularizers) so diagnostics and downstream methods can rely on a unified view.

---

## 10) Minimal Usage Sketch

```python
X = torch.randn(50_000, 64)

class MyAlgo(BaseClusteringAlgorithm):
    def _create_components(self):
        self.assignment_strategy = ...  # see interfaces (broken)
        self.update_strategy = ...
        self.initialization_strategy = ...
        self.convergence_criterion = ...
        self.objective = ...

    def _create_representations(self, data):
        # use initialization outputs inside this method as needed
        return [ ... for _ in range(self.n_clusters) ]

model = MyAlgo(n_clusters=8, max_iter=100, tol=1e-4, verbose=1).fit(X)
labels = model.predict(X)
centers = model.cluster_centers_
final_obj = model.inertia_
```

---

## 11) Related References (fill in as you share them)

* Interfaces: [interfaces.md](./interfaces.md) *(broken)*
* Data structures & state: [data\_structures.md](./data_structures.md) *(broken)*
* Example strategies:

  * Assignment: [assignments/](../assignments/README.md) *(broken)*
  * Updates: [updates/](../updates/README.md) *(broken)*
  * Initialization: [initialization/](../initialization/README.md) *(broken)*
  * Convergence: [utils/convergence.md](../utils/convergence.md) *(broken)*

---

*Last updated:* *(auto-generated draft from code signatures & comments)*
