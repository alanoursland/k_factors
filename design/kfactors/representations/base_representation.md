# BaseRepresentation — Design Guide

> A lightweight foundation for all cluster representations. It implements shared storage for core attributes (dimension, device, mean), common device transfer logic, and input shape validation. Concrete models (e.g., PPCA, Subspace) inherit from this class and implement the remaining `ClusterRepresentation` interface methods.

---

## Purpose

* Provide a **uniform base** that satisfies parts of the `ClusterRepresentation` contract shared by all models.
* Centralize:

  * Ambient **dimension** (`d`)
  * Target **device**
  * Cluster **mean** vector
* Offer safe utilities:

  * `.to(device)` deep move
  * `_check_points_shape(points)` validation

---

## Construction & Core State

```python
BaseRepresentation(dimension: int, device: torch.device)
```

* Initializes:

  * `_dimension: int` — ambient data dimension `d`.
  * `_device: torch.device` — where all tensors should live.
  * `_mean: Tensor[(d,)]` — initialized to zeros on `device`.

### Properties

* `dimension -> int` — read-only `d`.
* `device -> torch.device` — read-only current device.
* `mean -> Tensor[(d,)]` — get/set cluster centroid.

  * Setter enforces shape `(d,)` and moves to `_device`.

---

## Device Transfer

```python
to(device: torch.device) -> BaseRepresentation
```

* Creates a **new instance** of the same concrete class on `device`.
* Copies parameters by calling:

  * `params = self.get_parameters()`
  * `new_repr.set_parameters(params moved to device)`
* This implies subclasses must ensure:

  * `get_parameters()` returns **all** tensors necessary to fully define the representation.
  * `set_parameters(params)` can **reconstruct** internal state from that dict.

> Design intent: keep `.to(...)` generic and avoid subclass-specific transfer code. The correctness depends on complete/consistent implementations of `get_parameters`/`set_parameters` in subclasses.

---

## Input Validation

```python
_check_points_shape(points: Tensor)
```

* Ensures `points.ndim == 2` and `points.shape[1] == dimension`.
* Used by distance/likelihood/update routines in subclasses (e.g., PPCA), guarding against silent shape errors.

---

## Responsibilities Left to Subclasses

BaseRepresentation intentionally **does not** implement the full `ClusterRepresentation` contract. Subclasses must define:

* `distance_to_point(points, indices=None) -> Tensor[(n,)]`
* `update_from_points(points, weights=None, **kwargs) -> None`
* `get_parameters() -> Dict[str, Tensor]`
* `set_parameters(params: Dict[str, Tensor]) -> None`

…and may add representation-specific properties (e.g., `W`, `variance`, `basis`).

---

## Invariants & Expectations

* **Mean shape**: `(d,)` always.
* **Device consistency**: all tensors returned by `get_parameters()` should be on `self.device`. The base `.to(...)` will move them when constructing the new instance.
* **Reconstruction**: `set_parameters(get_parameters())` must be idempotent for the subclass (restores an equivalent state).

---

## Typical Usage Pattern

* Concrete representation (e.g., `PPCARepresentation`) extends `BaseRepresentation` to inherit device/mean handling and shape checks, then adds:

  * Its own parameter tensors (e.g., `W`, `variance`)
  * Efficient computations (distance, log-likelihood)
  * Update rules (batch ML/EM or sequential PCA)

---

## Why this design?

* **Simplicity**: keep the base lean; avoid assumptions about covariance/basis/etc.
* **Uniformity**: ensure every representation exposes the same **mean/device/dimension** semantics, reducing boilerplate in algorithms.
* **Safety**: centralize common pitfalls (device placement, shape checks) so algorithm code can assume consistency.

---

*In short: `BaseRepresentation` is the minimal, dependable scaffold for all cluster models — it standardizes the essentials and leaves the math-specific pieces to each subclass.*
