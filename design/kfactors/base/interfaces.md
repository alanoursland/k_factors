# Core Interfaces — Design Guide

> This document explains the contracts defined in `interfaces.py` for the K-Factors family. It focuses on **what each interface promises**, **how components interact**, and **shape/device conventions** expected by `BaseClusteringAlgorithm` and concrete algorithms (e.g., K-Factors).

---

## Design Goals

* **Composable**: Swap assignment, updates, objectives, and inits without touching the training loop.
* **Explicit contracts**: Each interface declares data shapes, return types, and side-effects.
* **Device-aware**: Everything must be movable to a target `torch.device`.
* **Hard/soft symmetry**: The loop supports both hard and soft assignments.

---

## ClusterRepresentation

Abstracts “what a cluster is.” Examples: a centroid (K-means), a centroid + basis (K-subspaces / K-Factors), a mean + covariance (GMM).

### Required API

```python
class ClusterRepresentation(ABC):
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor: ...
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None, **kwargs) -> None: ...
    def get_parameters(self) -> Dict[str, Tensor]: ...
    def set_parameters(self, params: Dict[str, Tensor]) -> None: ...
    @property
    def dimension(self) -> int: ...
    def to(self, device: torch.device) -> 'ClusterRepresentation': ...
```

### Contract & Expected Semantics

* **`distance_to_point(points, indices=None) -> Tensor[(n,)]`**

  * Returns a *per-point* cost consistent with the algorithm’s objective (e.g., squared residual for subspace models, negative log-likelihood for probabilistic models if used as “distance”).
  * Must be **batched** (no loops over points in Python if avoidable) and **device-correct**.

* **`update_from_points(points, weights=None, **kwargs) -> None`**

  * Updates the internal parameters of the representation *in place* from assigned points.
  * `weights=None` implies hard assignments; otherwise `weights.shape == (n,)` for soft.
  * Should accept optional hints via `**kwargs` (e.g., cached projections, stage index).

* **`get_parameters() -> Dict[str, Tensor]`**

  * **Must include**: `'mean': Tensor[(D,)]`.
  * **May include** (if model has them):

    * `'basis': Tensor[(D, r)]` — e.g., orthonormal columns for subspace/PPCA-like models.
    * `'covariance': Tensor[(D, D)]` or `'variance': Tensor[...]` — as appropriate.
  * All returned tensors should be on the representation’s device.

* **`set_parameters(params)`**

  * Must accept any dict produced by `get_parameters()` and fully restore the state.

* **`dimension: int`**

  * Ambient data dimension **D**. Used by the base class for checks and allocations.

* **`to(device)`**

  * Moves internal tensors to `device`. Return `self` to allow chaining.

### Invariants

* `get_parameters()['mean'].shape == (D,)` and matches `dimension`.
* If `'basis'` exists, its second dimension is the latent rank **r**; basis columns should be valid for projection in `distance_to_point`.
* Must be safe to call `distance_to_point` **before** any update (i.e., after initialization).

---

## AssignmentStrategy

Computes assignments from points to cluster representations.

### Required API

```python
class AssignmentStrategy(ABC):
    def compute_assignments(
        self,
        points: Tensor,
        representations: list[ClusterRepresentation],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]: ...

    @property
    def is_soft(self) -> bool: ...
```

### Contract

* **Input**: `points.shape == (n, D)`, `len(representations) == K`.
* **Output** (one of):

  * **Hard**: `assignments.shape == (n,)` with `Long` indices in `[0, K-1]`.
  * **Soft**: `assignments.shape == (n, K)` with non-negative weights (ideally rows sum to 1).
  * **With aux**: `(assignments, aux_info)` where `aux_info` may include caches like precomputed distances `(n, K)`, stage indices, penalties, etc.
* **`is_soft: bool`** governs how the base loop performs updates:

  * If `False`: the loop calls `get_cluster_indices(k)` and updates unweighted.
  * If `True` : the loop calls `get_cluster_weights(k)` and updates with weights.

### Notes

* If you return `(assignments, aux_info)`, the base loop forwards `aux_info` to the updater via `**aux_info`. Keep it **small** and **tensor-based** if possible.

---

## ParameterUpdater

Applies parameter updates to a single `ClusterRepresentation`.

### Required API

```python
class ParameterUpdater(ABC):
    def update(
        self,
        representation: ClusterRepresentation,
        points: Tensor,
        assignments: Tensor,   # None for hard; weights vector for soft; or strategy-specific
        **kwargs
    ) -> None: ...
```

### Contract

* **Representation**: updated **in place**.
* **Points**: typically the subset for a cluster (hard) or all points (soft) depending on how the base loop passes them.
* **Assignments**:

  * For **hard**: the base loop passes `None` and only the selected points.
  * For **soft**: the base loop passes a vector of weights (shape `(n_selected,)`).
  * Updaters may also accept richer `assignments` if their strategy requires it; align with the selected `AssignmentMatrix` semantics.
* **`**kwargs`**: receives any `aux_info` from the assignment stage (e.g., cached projections, current stage for sequential updates).

---

## DistanceMetric (Optional pluggable helper)

Encapsulates distance computation separate from `ClusterRepresentation`.

```python
class DistanceMetric(ABC):
    def compute(points: Tensor, representation: ClusterRepresentation, **kwargs) -> Tensor: ...
```

* Implement if you want to reuse metrics across multiple representations or to swap metrics without changing representations.

---

## InitializationStrategy

Produces initial cluster representations.

```python
class InitializationStrategy(ABC):
    def initialize(self, points: Tensor, n_clusters: int, **kwargs) -> list[ClusterRepresentation]: ...
```

### Contract

* Returns a list of **K** `ClusterRepresentation` instances, on the correct device, with at least `'mean'` valid and any essential parameters set so that `distance_to_point` can be called immediately.
* May accept `**kwargs` (e.g., random centers, k-means++ seeding, prior bases).

---

## ConvergenceCriterion

Decides when to stop the main loop (or a stage in staged algorithms).

```python
class ConvergenceCriterion(ABC):
    def __init__(self):
        self.history = []
    def check(self, current_state: Dict[str, Any]) -> bool: ...
    def reset(self): self.history = []
```

### Contract

* `check` receives a dictionary that may include: `iteration`, `objective`, raw `assignments`, `cluster_state`, and any algorithm metadata.
* Must be **deterministic** w\.r.t. supplied fields and maintain any internal counters in `self.history`.
* `reset()` clears prior state; base algorithms call this at the start of a run (or stage).

---

## ClusteringObjective

Computes the scalar objective used for progress and convergence.

```python
class ClusteringObjective(ABC):
    def compute(
        self,
        points: Tensor,
        representations: list[ClusterRepresentation],
        assignments: Tensor
    ) -> Tensor: ...
    @property
    def minimize(self) -> bool: ...
```

### Contract

* Returns a **scalar** tensor suitable for logging/comparison.
* Must align with the semantics used by `AssignmentStrategy` / `ParameterUpdater` so that the optimization direction makes sense.
* `minimize == True` implies lower is better; the base loop does not invert signs.

---

## Interactions & Data Shapes (Quick Reference)

* **n**: number of points; **D**: ambient dimension; **K**: number of clusters; **r**: latent rank (if any).
* `points`: `(n, D)` on the chosen device (`cpu`/`cuda`).
* `assignments`:

  * Hard: `(n,)` `Long`.
  * Soft: `(n, K)` `Float`, row-normalized recommended.
* `ClusterRepresentation.get_parameters()`:

  * Always: `'mean' -> (D,)`
  * Optional: `'basis' -> (D, r)`, `'covariance' -> (D, D)`, `'variance' -> (...)`
* **Device**:

  * All components must support `.to(device)` and return tensors on their device.
  * The base class will move `X` to a single device and expects downstream consistency.

---

## Minimal Example Sketches

### A centroid-only representation (K-means-like)

```python
class CentroidRep(ClusterRepresentation):
    def __init__(self, mean: Tensor):
        self._mean = mean  # (D,)

    def distance_to_point(self, points, indices=None):
        return ((points - self._mean) ** 2).sum(dim=1)

    def update_from_points(self, points, weights=None, **kwargs):
        if weights is None:
            self._mean = points.mean(dim=0)
        else:
            w = weights / (weights.sum() + 1e-12)
            self._mean = (w[:, None] * points).sum(dim=0)

    def get_parameters(self):
        return {'mean': self._mean}

    def set_parameters(self, params):
        self._mean = params['mean']

    @property
    def dimension(self):
        return self._mean.numel()

    def to(self, device):
        self._mean = self._mean.to(device)
        return self
```

### An assignment strategy (hard, argmin distance)

```python
class HardArgminAssignment(AssignmentStrategy):
    @property
    def is_soft(self): return False

    def compute_assignments(self, points, reps, **kwargs):
        # stack distances: (n, K)
        dists = torch.stack([rep.distance_to_point(points) for rep in reps], dim=1)
        return dists.argmin(dim=1)  # (n,)
```

---

## Validation Checklist for Implementers

* [ ] `ClusterRepresentation.get_parameters()` returns at least `'mean'` `(D,)`.
* [ ] `ClusterRepresentation.to(device)` moves **all** tensors and returns `self`.
* [ ] `AssignmentStrategy.is_soft` matches the shape of `assignments` you return.
* [ ] `ParameterUpdater.update(...)` accepts both hard (`assignments=None`) and soft cases as used by the base loop, or document stricter expectations.
* [ ] `ClusteringObjective.minimize` matches the intended optimization direction.
* [ ] All tensor returns/shapes are **batched** and **device-consistent**.

---

*This guide captures the intended contracts from `interfaces.py` so downstream algorithms (like K-Factors) and the base loop can interoperate predictably.*
