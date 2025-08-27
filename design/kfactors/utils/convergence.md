# Convergence Criteria — Design Guide

> This module provides **stopping conditions** for clustering algorithms.
> Each criterion implements `ConvergenceCriterion.check(state: Dict[str, Any]) -> bool`, returning whether to halt optimization based on different signals.

---

## Purpose in K-Factors

K-Factors (and other algorithms built on the base framework) iteratively:

1. Assign points,
2. Update cluster parameters,
3. Compute an objective,
4. **Check convergence**.

This module defines alternative criteria for step (4). They allow flexibility: algorithms can stop when assignments stabilize, when objective values plateau, when cluster parameters stop moving, or some combination.

---

## Criteria Overview

### 1. **ChangeInAssignments**

Converges when **few points change clusters** across iterations.

* **Parameters**

  * `min_change_fraction`: threshold fraction of points that must change (default `1e-4`).
  * `patience`: number of consecutive stable iterations required.

* **Logic**

  * Compares current assignments (hard labels) with previous.
  * Computes:

    $$
    \text{change\_fraction} = \frac{\#\{i: a^{(t)}_i \neq a^{(t-1)}_i\}}{n}.
    $$
  * If `change_fraction < min_change_fraction` for `patience` iterations in a row → converge.

* **History Tracked**

  * Iteration, `n_changed`, `change_fraction`.

---

### 2. **ChangeInObjective**

Converges when the **objective function stabilizes**.

* **Parameters**

  * `rel_tol`: relative tolerance (default `1e-4`).
  * `abs_tol`: absolute tolerance (default `1e-8`).
  * `patience`: required consecutive stable iterations.

* **Logic**

  * Compares current objective value with previous.
  * Computes absolute and relative changes:

    $$
    \Delta = |f^{(t)} - f^{(t-1)}|,\quad \Delta_{\text{rel}} = \frac{\Delta}{|f^{(t-1)}|}
    $$
  * Converge if `Δ < abs_tol` or `Δ_rel < rel_tol`, sustained for `patience` iterations.

* **History Tracked**

  * Iteration, current objective, `abs_change`, `rel_change`.

---

### 3. **ParameterChange**

Converges when **cluster parameters stop changing**.

* **Parameters**

  * `tol`: relative Frobenius norm tolerance (default `1e-6`).
  * `patience`: consecutive stable iterations required.
  * `parameter`: which parameter to monitor (`'mean'`, `'basis'`, `'covariance'`).

* **Logic**

  * Extracts the chosen tensor from `cluster_state`.
  * Computes relative change:

    $$
    \text{rel\_change} = \frac{||\theta^{(t)} - \theta^{(t-1)}||_F}{||\theta^{(t-1)}||_F}.
    $$
  * Converge if `rel_change < tol` for `patience` iterations.

* **History Tracked**

  * Iteration, parameter change value.

---

### 4. **CombinedCriterion**

Combines multiple criteria with **AND** / **OR** logic.

* **Parameters**

  * `criteria`: list of `ConvergenceCriterion` instances.
  * `mode`: `"any"` (stop if any converge) or `"all"` (stop only if all converge).

* **Logic**

  * Evaluates all criteria on the current state.
  * Aggregates results via logical `any` or `all`.

* **History Tracked**

  * Iteration, individual criterion results, final decision.

* **Reset**

  * Resets self and all sub-criteria histories.

---

### 5. **MaxIterations**

Degenerate criterion: always returns `False`.

* The outer algorithm enforces a hard cap on iterations (`max_iter`), so this criterion exists mostly for consistency.

---

## Shared Features

* All classes extend `ConvergenceCriterion`, which provides:

  * `.history`: list of diagnostic dicts.
  * `.reset()`: clears history.

* **Patience mechanism**: avoids premature convergence due to fluctuations. Many criteria require stability across multiple iterations.

* **Fallbacks**: If a requested parameter (`basis`, `covariance`) is missing in `ParameterChange`, it defaults to monitoring `means`.

---

## When to Use Which

* **Assignments-based**: Works best for k-means-like hard clustering where assignments stabilize quickly.
* **Objective-based**: Natural for EM or variational methods where objective monotonically improves.
* **Parameter-based**: Suited for models with rich cluster structure (subspaces, covariance), especially when assignments may remain noisy but parameters stabilize.
* **Combined**: In safety-critical algorithms, combine multiple criteria (`assignments OR objective`) for robustness.

---

## Example: Integrating into K-Factors

```python
criterion = CombinedCriterion([
    ChangeInAssignments(min_change_fraction=1e-3, patience=2),
    ChangeInObjective(rel_tol=1e-4, patience=3)
], mode="any")

algorithm.convergence_criterion = criterion
```

This setup halts if either assignments stabilize for 2 iterations **or** objective stops improving for 3 iterations.

---

*These convergence criteria provide flexible, pluggable stopping rules for clustering algorithms. They balance stability, robustness, and computational efficiency, and can be combined to suit different algorithmic needs.*
