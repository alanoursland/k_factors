# K-Factors Test Plan

This plan defines a **lean but meaningful** suite of unit and integration tests for the K-Factors implementation. It aims to catch real bugs, validate core behavior on toy data, and record wall-clock timings—without going overboard.

---

## Goals

* Verify that K-Factors **learns local bases** that align with dominant directions of the data (up to sign/scale/permutation).
* Check that the **penalty weighting** and **sequential extraction** behave sensibly across stages.
* Ensure **convergence behavior** is reasonable and the implementation is **numerically stable**.
* Capture **timing** for key scenarios to watch performance drift.

## Scope

* **In scope:** K-Factors pipeline and the components it uses (`DirectionTracker`, `PenalizedAssignment`, `SequentialPCAUpdater`, `PPCARepresentation` aspects exercised by K-Factors), basic linalg helpers as needed.
* **Out of scope (for now):** Other clustering algorithms, full PPCA likelihood tests, exhaustive linalg validation, GPU-specific perf targets.

---

## Common Fixtures & Helpers (conceptual)

* **Seeded RNG:** Fix `random_state` and NumPy/Torch seeds for deterministic tests.
* **Metrics:**

  * **Abs-cosine similarity** between learned directions and ground truth.
  * **Cluster separation accuracy** (permutation-invariant 0/1 correctness on labeled toy sets).
  * **Objective trace** monotonicity (non-increasing if minimize).
* **Timing capture:** Use `time.perf_counter()` around fit blocks and print/log durations (or mark with pytest to show durations).

---

## Unit Tests

### U1 — `DirectionTracker` stores and returns weighted penalties

**Purpose:** Validate weighted product and weighted mean penalty computations.

* Build a small history for a single point:

  * Two stored unit directions `d0 = ex`, `d1 = ey`, with weights `a0=1.0`, `a1=1.0`.
  * Test candidate `u = (ex + ey)/√2`.
* **Expectations:**

  * **product mode:** penalty ≈ $(1−|cos|)^2$ = $(1−0.7071)^2 ≈ 0.0858$.
  * **sum/mean mode (weighted mean of overlaps):** penalty ≈ $1 − (0.7071+0.7071)/2 ≈ 0.2929$.
* Also verify:

  * If a weight is **0**, it doesn’t affect the penalty.
  * Returned values are clamped to **\[0,1]**.
  * **Batch** and **single-point** methods agree for equivalent inputs.

### U2 — `PenalizedAssignment` produces the right **per-point per-cluster** weights

**Purpose:** Given `current_directions` and a `DirectionTracker` with known history/weights, verify the `(n,K)` penalty matrix matches hand calculations in simple 2D/3D setups.

* Scenario: three candidate directions `[ex, ey, (ex+ey)/√2]`, one point with prior claims on `ex` (weight 1) and `ey` (weight 1).
* **Expectations:** penalties for `[ex, ey, diag]` ≈ `[0, 0, 0.0858]` (product mode); shape and dtype correct.

### U3 — `SequentialPCAUpdater` extracts the top residual direction

**Purpose:** With points along a noisy line in 3D, confirm the updater’s **first stage** direction aligns with the line.

* Input: cluster points near `ex`, updater at `stage=0`.
* **Expectation:** learned column direction (after normalizing) has abs-cosine > 0.95 with `ex`.
* **Note:** We don’t require unit norm; normalize in the check.

### U4 — Weighted update actually uses **per-point weights**

**Purpose:** Show that changing `assignment_weights` affects the learned direction.

* Build points that are a mix of `ex` and `ey`. Run **two** updates:

  * Case A: all weights 1 → result close to combined PC.
  * Case B: down-weight the `ey` points → result closer to `ex`.
* **Expectation:** abs-cosine to `ex` in Case B > Case A.

### U5 — Convergence criterion records history and stabilizes

**Purpose:** Feed a decreasing objective sequence and confirm `ChangeInObjective` sets `converged=True` after `patience` is met; verify `history` entries exist and contain expected fields.

### U6 — Data validation & device routing

**Purpose:** `_validate_data` enforces 2D float tensors, moves to device; `get_hard/get_soft` shape behavior is consistent.

* Pass list/np array → get float32 tensor on target device.
* `AssignmentMatrix` conversions: hard→soft (one-hot), soft→hard (argmax), and counts.

---

## Integration Tests

### I1 — **Two 1D lines in 3D** (the passing test)

**Purpose:** End-to-end sanity on the simplest shape.

* Data: 200 points near `ex`, 200 near `ey`.
* Model: `K=2, R=1`, default penalty (product).
* **Checks:**

  * Directions (normalize before compare) align with `{ex, ey}`; permutation/sign invariant match score > 1.8.
  * Label separation accuracy > 0.9 (permutation-safe).
  * **Timing:** record wall time for `fit`.

### I2 — **Two anisotropic Gaussians** with different covariances

**Purpose:** Validate learning of **local** bases; no cross-cluster orthogonality assumed.

* Cluster A: covariance diag$[3.0, 0.3, 0.1]$ rotated by some angle.
* Cluster B: covariance diag$[2.5, 0.2, 0.1]$ rotated by a different angle.
* Model: `K=2, R=1` initially; optionally repeat with `R=2`.
* **Checks:**

  * Per-cluster direction aligns with **local top PC** (computed via numpy PCA on the cluster’s true generating covariance or on the generated samples with labels).
  * Acc > 0.85 (easier to pass with R=1).
  * Timing recorded.

### I3 — **Sequential extraction** within cluster (R=2) on a plane in 3D

**Purpose:** Confirm stage-wise residual PCA.

* Data: one cluster on a 2D plane spanned by `ex, ey`, plus a second cluster on a different 2D plane (to exercise `K=2, R=2`).
* **Checks (per cluster):**

  * Stage-1 direction aligns with the top plane axis (>0.9 abs-cosine).
  * Stage-2 direction lies **in the plane** and is approximately orthogonal to stage-1 (**within the cluster**, not across clusters); abs-cosine(stage1, stage2) < 0.2.
  * Timing recorded.

### I4 — **Penalty heuristic behavior** on staged claims

**Purpose:** Show the weighted product heuristic “eats credit” as designed.

* Construct one point with prior claims on two orthogonal directions with weights 1, and test candidate diagonal:

  * Verify stage weights pulled from `aux_info['penalties']` match \~0.0858 for the diagonal.
* Run a tiny fit (e.g., `n=10`) over 2–3 stages and assert that the **effective sample size** (sum of stage weights) for candidates aligned with already-claimed directions is smaller than for orthogonal candidates.

### I5 — **Label permutation & sign robustness**

**Purpose:** Ensure metrics and checks are permutation/sign invariant.

* Re-run I1/I2 with clusters swapped in data ordering; confirm tests still pass.

### I6 — **Objective trace sanity**

**Purpose:** Ensure we don’t regress to “objective increases wildly.”

* Capture objective per inner iteration via `model.history_` (or stage history for sequential loop).
* **Check:** for minimize objectives, last objective ≤ first objective; within a stage, it should be non-increasing or at least not increase by more than a small epsilon due to numerical noise.
* Timing recorded.

### I7 — **Determinism with fixed seed**

**Purpose:** With `random_state` fixed, two runs yield the same directions/labels.

* Compare normalized bases and labels (after permutation alignment). Tolerate tiny float diffs.

### I8 — **Edge robustness**

**Purpose:** Guard obvious edge failures.

* Very small cluster: e.g., `n=6`, `K=2`, `R=1`.
* High-dim, low-n: e.g., `n=100`, `d=200`, `K=3`, `R=2` (light).
* Colinear points all on one line: algorithm should not crash; stage-2 direction may be unstable but convergence shouldn’t explode.
* Zero-variance feature column in X: no crash.
* For each: **no exceptions**, fit completes, timing recorded.

---

## Pass Criteria (suggested thresholds)

* **Direction alignment:** abs-cosine > 0.9 on relevant axes (permutation/sign invariant).
* **Cluster accuracy:** > 0.9 on easy synthetic (I1), > 0.85 on anisotropic (I2).
* **Intra-cluster stage orthogonality:** < 0.2 abs-cosine between stage-1 and stage-2 directions (I3).
* **Penalty numeric bounds:** penalties in \[0,1]; expected heuristic values within ±0.02 absolute tolerance on simple cases (U1/U2/I4).
* **Determinism:** identical outcomes (within 1e-6 for directions after alignment; identical labels).
* **Objective sanity:** non-increasing across iterations within minor epsilon.

---

## Timing Collection

For each integration test, record:

* **n, d, K, R**, and **wall clock fit time** (seconds) using `perf_counter()`.
* Print to stdout (or `pytest` `-vv -s` to surface).
  Optionally, mark tests with `@pytest.mark.slow` when sizes increase, and run them locally/CI nightly.

Example sizes to time:

* I1: `n=400, d=3, K=2, R=1`.
* I2: `n=1000, d=10, K=2, R=1/2`.
* I3: `n=600, d=3, K=2, R=2`.
* I8 high-dim: `n=100, d=200, K=3, R=2` (smoke timing).

We’re not asserting perf thresholds yet—just **recording** to detect regression over time.

---

## Failure Triage Notes

* If **alignment** fails but shapes pass, inspect:

  * Are weights actually passed into the updater?
  * Are claimed weights recorded at stage end?
  * Are candidate directions unit-normalized before cosine penalties?
* If **accuracy** fails but directions look okay:

  * Check assignment step vs update step consistency and distance metric.
* If **objective** increases erratically:

  * Verify convergence criterion setup per stage; inspect residual covariance computations and weighting.

---

## Coverage Map (high-level)

* `DirectionTracker` → U1, U2, I4
* `PenalizedAssignment` → U2, I4
* `SequentialPCAUpdater` → U3, U4, I3
* `PPCARepresentation` (as used) → U3, U4, I3
* `BaseClusteringAlgorithm` loop & convergence → I1–I3, I6, I7, I8
* Utility behaviors (validation, conversions) → U5, U6

---
===============================================================================

# Implementation Plan (Order of Work)

## Phase 0 — Test scaffolding (tiny, one-time)

**Goal:** Make later tests short, readable, and deterministic.

1. **Shared fixtures & config**

* Seed fixture (`numpy`, `torch`, Python `random`) and `random_state` param.
* Force CPU (or set CUDA deterministic flags if you’ll run on GPU).
* Optional: `torch.set_num_threads(1)` to reduce flakiness.

2. **Helpers (in `tests/utils.py`)**

* `unit(v)`: normalize vector.
* `abs_cos(u, v)`: absolute cosine similarity.
* `perm_invariant_axis_match(B, G)`: returns best sum of |cos| under permutation.
* `perm_invariant_accuracy(y_pred, n_first)`: returns best 0/1 accuracy given known split point.
* `time_block(label)`: context manager to print wall-clock time.
* Tiny synthetic generators:

  * `make_two_lines_3d(n=200, noise=0.1)`
  * `make_aniso_gaussians(d=10, rotations=..., covs=...)`
  * `make_two_planes_3d(...)`

> **Gate to proceed:** helpers import cleanly; a dummy test can use `time_block()` and print a duration.

---

## Phase 1 — Core unit tests (low effort, high confidence)

**Why first:** They validate the building blocks in isolation.

1. **U1 — DirectionTracker penalties**

* Single-point, hand-check values for product & mean modes.
* Also check zero-weight claim is ignored; batch vs single match.

2. **U3 — SequentialPCAUpdater extracts top residual**

* One cluster ≈ line in 3D (stage 0). Assert |cos| > 0.95 to true axis.

3. **U4 — Weighted update respects weights**

* Mixed ex/ey points; show direction moves toward ex when ey points are down-weighted.

4. **U2 — PenalizedAssignment matrix**

* Small (n=1, K=3) example; verify penalties ≈ expected numbers.

5. **U5 — Convergence criterion behavior**

* Feed a synthetic objective sequence to `ChangeInObjective`; confirm patience/threshold logic.

6. **U6 — Validation & assignment matrix conversions**

* `_validate_data` on list/np; `AssignmentMatrix` hard↔soft consistency.

> **Gate to proceed:** all Phase 1 tests green on CPU locally.

---

## Phase 2 — First integration pass (the “smoke that matters”)

**Why now:** With units solid, verify the end-to-end happy path.

1. **I1 — Two 1D lines in 3D**

* Use your passing test (alignment + accuracy, permutation/sign invariant).
* Wrap `fit` in `time_block("I1")` and print n/d/K/R + seconds.

2. **I5 — Label permutation & sign robustness**

* Same data as I1 but with rows swapped; ensure I1 assertions still pass.

3. **I6 — Objective trace sanity**

* Capture objective from `model.history_` (or stage records). Assert last ≤ first (allow tiny epsilon). Print iteration count + time.

> **Gate to proceed:** I1/I5/I6 green and timing prints look sane.

---

## Phase 3 — Broaden coverage (local bases & stages)

**Purpose:** Validate “local basis” learning and sequential extraction across clusters.

1. **I2 — Two anisotropic Gaussians (d=10)**

* Compute per-cluster empirical PCA (using true labels from generator).
* Check learned direction aligns with top PC per cluster (|cos| > 0.9).
* Accuracy > 0.85. Record time.

2. **I3 — Sequential extraction on planes (K=2, R=2)**

* For each cluster: stage-1 aligns with top plane axis; stage-2 is in-plane and near-orthogonal to stage-1 (within cluster). Record time.

> **Gate to proceed:** I2/I3 green; no flakiness across 3 seeded runs.

---

## Phase 4 — Penalty behavior as seen end-to-end

**Purpose:** Ensure the heuristic “eats credit” in actual runs.

1. **I4 — Penalty heuristic behavior**

* Tiny dataset, 2–3 stages. From `aux_info['penalties']`/gathered stage weights, verify:

  * candidate aligned with claimed directions gets smaller effective weight than an orthogonal candidate
  * numeric value roughly matches the simple analytic case (tolerance ±0.02).
* Record time.

> **Gate to proceed:** I4 matches expectations; log print shows effective sample sizes per cluster and stage.

---

## Phase 5 — Determinism & edges

**Purpose:** Lock down stability and guard simple edge cases.

1. **I7 — Determinism**

* With fixed seed & same device, two runs produce same labels (after permutation alignment) and same normalized bases within 1e-6.

2. **I8 — Edge robustness**

* Small n (6), high-dim low-n (n=100, d=200, K=3, R=2), colinear points, zero-variance feature.
* Assertions: **no exception**, fit completes, objective finite, timing recorded. (No strict accuracy thresholds here.)

> **Gate to finish:** I7/I8 green; durations logged.

---

## Phase 6 — Markers, CI hygiene, and “slow” policy

* Mark heavier tests (`I2`, `I3`, high-dim `I8`) with `@pytest.mark.slow`.
* Default CI runs **Phase 1–2** + **I4**. Nightly (or local) runs include **slow**.
* Print timing summaries only on success to keep CI noise low.

---

## Practical sequencing checklist

1. **Scaffold** (Phase 0)
2. **U1 → U3 → U4 → U2 → U5 → U6** (Phase 1)
3. **I1 → I5 → I6** (Phase 2)
4. **I2 → I3** (Phase 3)
5. **I4** (Phase 4)
6. **I7 → I8** (Phase 5)
7. **Marks & CI** (Phase 6)

Each step should be <\~1–2 hours of work; stop after each gate, run green, and commit. This order front-loads the biggest de-riskers (penalty math + updater behavior) before spending time on larger, slower integration scenarios, and gets you useful **timing baselines** early so you can watch for performance drift.

---
===============================================================================

# Phase 0 scaffolding plan

Here’s a concrete, lightweight **Phase 0 scaffolding plan**—exact files to add, what each will contain, and the (very small) smoke tests to confirm the scaffolding works before we start real tests.

---

# Directory layout

```
tests/
  conftest.py
  utils.py
  data_gen.py
  test_scaffold_smoke.py
```

No `__init__.py` needed under `tests/`.

---

# File-by-file plan

## 1) `tests/conftest.py`

**Purpose:** global fixtures for seeding, device choice, and thread settings. Keep them tiny and predictable.

### Fixtures to implement

* `seed_all` (session-scoped)

  * Sets deterministic seeds for `random`, `numpy`, and `torch` (CPU/GPU if present).
  * Seed value: from env var `TEST_RANDOM_SEED` (default `1337`).
  * Optionally set `torch.use_deterministic_algorithms(False)` and `torch.backends.cudnn.benchmark = False` for consistency.
* `rng`

  * Returns a `numpy.random.Generator` using `PCG64` seeded from `seed_all`.
* `torch_device`

  * Returns `torch.device('cpu')` (we’ll keep Phase 0 strictly on CPU).
* `set_torch_threads` (session-scoped, autouse)

  * Calls `torch.set_num_threads(1)` to reduce flakiness and stabilize timings.

> These fixtures are just plumbing; they don’t run KFactors yet.

---

## 2) `tests/utils.py`

**Purpose:** small, reusable helpers used by later tests. Implement once, reuse everywhere.

### Functions to implement

* `unit(v) -> same type as input`

  * Returns the unit-normalized vector; supports `np.ndarray` and `torch.Tensor`.
* `abs_cos(u, v) -> float`

  * Absolute cosine similarity; accepts numpy or torch vectors.
* `perm_invariant_axis_match(B, G) -> (best_score: float, best_perm: tuple[int, ...])`

  * `B`: array of learned unit vectors (shape `(k, d)`).
  * `G`: array of ground-truth unit vectors `(k, d)`.
  * Computes the best sum of absolute cosines under all permutations of `G`; returns the score and the arg permutation.
* `perm_invariant_accuracy(y_pred: np.ndarray, split_index: int) -> float`

  * For synthetic “first n are cluster 0, rest cluster 1” datasets; returns best accuracy over label swaps.
* `time_block(label: str, meta: dict | None = None)`

  * Context manager that prints `label`, optional `meta` (e.g., `{"n":..., "d":..., "K":..., "R":...}`), and wall-clock seconds using `time.perf_counter()`.
  * Used to record timings in later tests.
* (Optional) `print_timing(label: str, seconds: float, **meta)`

  * Helper if we want non-context-manager prints as well.

**Notes**

* Keep these pure and dependency-light; they’ll be imported across all test phases.

---

## 3) `tests/data_gen.py`

**Purpose:** tiny synthetic-data generators we’ll reuse in Phase 1–5 tests. For Phase 0 we only sketch signatures and docstrings; the actual logic can come in Phase 1.

### Functions to implement (signatures + docstrings now; full bodies in Phase 1)

* `make_two_lines_3d(n_per: int = 200, noise: float = 0.1, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

  * Returns `(X, y, G)` where:

    * `X`: `(2*n_per, 3)` points; first cluster along `ex`, second along `ey` with noise.
    * `y`: `(2*n_per,)` true labels `[0]*n_per + [1]*n_per`.
    * `G`: `(2, 3)` ground-truth unit directions (`ex`, `ey`).
* `make_aniso_gaussians(n_per: int, d: int, covA: np.ndarray, covB: np.ndarray, rotA: np.ndarray | None, rotB: np.ndarray | None, seed: int | None = None) -> tuple[X, y, (GA, GB)]`

  * Two anisotropic Gaussian clusters in `d` dims; returns data/labels and per-cluster ground-truth top PCs (or placeholders—computed in Phase 2).
* `make_two_planes_3d(n_per: int, noise: float, seed: int | None = None) -> tuple[X, y, (G1, G2)]`

  * Two clusters, each on a 2D plane in 3D; returns plane bases for ground truth (2×3 each).

**Phase 0 deliverable:** the functions exist with docstrings and `raise NotImplementedError` so imports succeed and later PRs can fill them in.

---

## 4) `tests/test_scaffold_smoke.py`

**Purpose:** tiny tests to ensure the scaffolding loads and our helpers work. No algorithm logic yet.

### Tests to write

* `test_utils_imports(seed_all)`

  * Imports `utils` and `data_gen` successfully.
  * Asserts the helper names exist (e.g., `hasattr(utils, "time_block")` etc.).
* `test_time_block_prints_duration(capsys)`

  * Uses `with time_block("noop"):` and a trivial sleep or loop.
  * Captures stdout and asserts it contains `noop` and a floating-point seconds value.
* `test_seed_consistency_rng(seed_all, rng)`

  * Draw a small array twice from `rng` in separate calls (re-seeding via `seed_all` fixture between tests happens automatically due to session scope—so here just ensure the same generator instance is used deterministically).
  * Assert arrays are equal (bitwise).
* `test_perm_invariant_axis_match_tiny()`

  * Construct `B = [[1,0,0],[0,1,0]]`, `G = [[0,1,0],[1,0,0]]`; expect best score ≈ 2.0 and permutation `(1,0)`.
* `test_perm_invariant_accuracy_tiny()`

  * `y_pred = [0,0,0,1,1,1]`, `split_index=3` ⇒ accuracy = 1.0;
    `y_pred = [1,1,0,0,0,1]` ⇒ best accuracy = 2/3.

These confirm: imports work, timing prints, permutation helpers behave, and our seed fixture yields deterministic behavior.

---

## What we’ll **not** do in Phase 0

* No `KFactors` imports or fits yet.
* No synthetic data generation logic (only function shells with docstrings).
* No assertions on algorithm outputs.

---

## Definition of done (Phase 0)

* All four files exist and import cleanly.
* `pytest -q` runs and passes the scaffold tests.
* The `time_block` helper visibly prints durations in at least one test (so we know the timing hook works before using it on model fits).

---
===============================================================================
# Phase 1 — Core Unit Tests (files, contents, assertions)

## Directory layout

```
tests/
  unit/
    test_direction_tracker.py
    test_sequential_pca_updater.py
    test_weighted_updates.py
    test_penalized_assignment.py
    test_convergence.py
    test_validation_assignment_matrix.py
```

All tests run on CPU, use the Phase-0 fixtures (`seed_all`, `rng`, `torch_device`) and helpers (`utils.unit`, `utils.abs_cos`, `utils.time_block`).

---

## 1) `tests/unit/test_direction_tracker.py`  (U1 — penalties)

**Imports**

* `torch`
* `numpy as np`
* `from kfactors.base.data_structures import DirectionTracker`
* `from tests.utils import abs_cos, unit`

**Tests**

1. `test_penalty_product_simple_orth_parallel()`

   * Build a `DirectionTracker(n_points=1, n_clusters=3, device=cpu)`.
   * Add history for point 0: `(e_x, weight=1.0)` and `(e_y, weight=1.0)`.
   * Candidates:

     * `u_parallel = e_x` → product penalty: `(1 - 1)*(1 - 0) = 0.0`.
     * `u_orth = e_z` → product penalty: `(1 - 0)*(1 - 0) = 1.0`.
     * `u_diag = (e_x + e_y)/√2` → product penalty: `(1 - √½)^2 ≈ 0.085786…`.
   * Assert each penalty within `±1e-6` of expected.

2. `test_penalty_mean_normalization()`

   * Same history; “mean” mode (i.e., `1 - (Σ w_i |cos|)/(Σ w_i)`).
   * Check:

     * `u_parallel = e_x` → `1 - (1/2) = 0.5`.
     * `u_orth = e_z` → `1 - 0 = 1.0`.
     * `u_diag` → `1 - (√½+√½)/2 = 1 - √½ ≈ 0.292893…`.
   * Assert within `±1e-6`.

3. `test_zero_weight_claim_ignored()`

   * History: `(e_x, weight=0.0)` only.
   * Any candidate → penalty should be `1.0` in both modes.
   * Assert exactly `1.0`.

4. `test_batch_matches_singleton()`

   * History: `(e_x, 1.0)` and `(e_y, 0.5)`.
   * Build `test_directions` of shape `(n=1, K=3, d=3)` with candidates `[e_x, e_y, (e_x+e_y)/√2]`.
   * Compare `compute_penalty_batch()[0]` with three calls to `compute_penalty(...)` (same `penalty_type='product'` then `'sum'`).
   * Assert `np.allclose` within `1e-7`.

---

## 2) `tests/unit/test_sequential_pca_updater.py`  (U3 — top residual)

**Imports**

* `torch`, `numpy as np`
* `from kfactors.updates.sequential_pca import SequentialPCAUpdater`
* `from kfactors.representations.subspace import SubspaceRepresentation`
* `from tests.utils import abs_cos, unit`

**Tests**

1. `test_stage0_extracts_top_axis_subspace()`

   * Generate 1D line in 3D: points `~ α * e_x + ε`, `α ~ N(0,1)`, `ε ~ 0.05*N(0,I)`, size \~ 400.
   * Create `SubspaceRepresentation(dimension=3, subspace_dim=1, device=cpu)` with default mean.
   * Call `SequentialPCAUpdater().update(repr, points, current_stage=0)`.
   * Assert `abs_cos(repr.basis[:,0], e_x) > 0.95`.

2. (Optional) `test_stage0_extracts_top_axis_ppca()`

   * Same data but with `PPCARepresentation(dimension=3, latent_dim=1, device=cpu)`.
   * After update, compare direction via `repr.W[:,0] / ||.||`.
   * Assert `abs_cos(...) > 0.95`.

---

## 3) `tests/unit/test_weighted_updates.py`  (U4 — weights matter)

**Imports**

* `torch`, `numpy as np`
* `from kfactors.representations.subspace import SubspaceRepresentation`
* `from kfactors.representations.ppca import PPCARepresentation`
* `from tests.utils import abs_cos, unit`

**Tests**

1. `test_subspace_update_weight_biases_direction()`

   * Build two equal-sized clouds:

     * Cluster A along `e_x` (n=200), Cluster B along `e_y` (n=200), noise 0.05.
   * Combine to one set `X`.
   * Case A (bias to `e_x`):

     * Weights: ones for A rows; very small (e.g., 0.05) for B rows.
     * `SubspaceRepresentation(d=3, r=1)`. Call `update_from_points(X, weights)`.
     * Assert `abs_cos(basis[:,0], e_x) > 0.9`.
   * Case B (bias to `e_y`):

     * Swap weights (A small, B ones).
     * Assert `abs_cos(basis[:,0], e_y) > 0.9`.

2. `test_ppca_update_weight_biases_direction()`

   * Same setup with `PPCARepresentation(d=3, r=1)`.
   * After `update_from_points(X, weights)`, normalize `W[:,0]` and assert alignment as above.
   * Optionally assert noise `variance > 0`.

---

## 4) `tests/unit/test_penalized_assignment.py`  (U2 — penalties via assignment)

**Imports**

* `torch`, `numpy as np`
* `from kfactors.assignments.penalized import PenalizedAssignment`
* `from kfactors.base.data_structures import DirectionTracker`
* `from kfactors.representations.subspace import SubspaceRepresentation`

**Tests**

1. `test_penalties_reflected_in_aux_info_product()`

   * `n=1`, `K=3`, `d=3`. Make three `SubspaceRepresentation(d=3, r=1)` with bases:

     * f0 = `e_x`, f1 = `e_y`, f2 = `(e_x + e_y)/√2`.
     * Means = 0.
   * Build `DirectionTracker` with history for the single point: `(e_x, 1.0)` and `(e_y, 1.0)`.
   * Call `PenalizedAssignment(penalty_type='product', penalty_weight=1.0).compute_assignments(points=[0,0,0])`.
   * Read `aux_info['penalties'][0]`, expect approximately:

     * f0 ≈ 0.0
     * f1 ≈ 0.0
     * f2 ≈ `(1 - √½)^2 ≈ 0.085786…`
   * Assert with tolerances `±1e-6`.

2. `test_penalties_reflected_in_aux_info_mean()`

   * Repeat with `penalty_type='sum'` (mean form).
   * Expect:

     * f0 ≈ 0.5
     * f1 ≈ 0.5
     * f2 ≈ `1 - √½ ≈ 0.292893…`.

> Note: We’re checking the **penalty** values surfaced through `aux_info['penalties']`, not any “penalized distances”.

---

## 5) `tests/unit/test_convergence.py`  (U5 — criterion behavior)

**Imports**

* `from kfactors.utils.convergence import ChangeInObjective, ChangeInAssignments, ParameterChange`

**Tests**

1. `test_change_in_objective_patience_and_thresholds()`

   * Create `ChangeInObjective(rel_tol=1e-3, abs_tol=1e-8, patience=2)`.
   * Feed sequence of states with `objective`: `[100.0, 99.9, 99.89995, 99.89990]`.

     * First call returns `False` (baseline established).
     * Second: relative change ≈ 0.001 → **just at** tolerance (decide expected: treat “< rel\_tol” as stable; “==” counts as not stable → we’ll use slightly smaller change to be safe).
     * Use tiny decrements to trigger two consecutive “stable” detections → expect `True` on the 4th call.
   * Assert internal `history` length increments, `_stable_count` behavior (via result pattern).

2. (Optional) `test_change_in_assignments_fraction()`

   * If you want: craft two hard label vectors that differ by < threshold fraction (e.g., 0.0009) for two consecutive calls → expect `True` with `patience=2`.

---

## 6) `tests/unit/test_validation_assignment_matrix.py`  (U6 — validation & conversions)

**Imports**

* `torch`, `numpy as np`
* `from kfactors.base.clustering_base import BaseClusteringAlgorithm` (if accessible) or `from kfactors.algorithms import KFactors` to reach `_validate_data`
* `from kfactors.base.data_structures import AssignmentMatrix`

**Tests**

1. `test_validate_data_numpy_and_list_to_tensor()`

   * Create `KFactors(n_clusters=2, n_components=1, random_state=0, device=cpu)` just to access `_validate_data`.
   * Pass a Python list of lists and a `np.ndarray` (float64).
   * Assert returned `torch.Tensor` is `float32`, `2D`, and on `cpu`.

2. `test_assignment_matrix_hard_to_soft_and_back()`

   * Hard: `assign = [0,1,0,1]`, `K=2`.
   * `AssignmentMatrix(assign, K, is_soft=False)` → `get_soft()` → expect one-hot `(n,2)`.
   * Construct soft AM with that soft matrix → `get_hard()` → `array([0,1,0,1])`.
   * `count_per_cluster()` equals `[2.0, 2.0]` for hard / equals sums for soft.

3. `test_assignment_matrix_get_cluster_indices_threshold_behavior()`

   * Soft with responsibilities: e.g., `[[0.6,0.4],[0.4,0.6],[0.51,0.49],[0.49,0.51]]`.
   * `get_cluster_indices(0)` returns indices where column 0 > 0.5 → `{0,2}`.
   * `get_cluster_indices(1)` → `{1,3}`.

---

## Gate to proceed

* All Phase 1 tests pass locally on CPU (`pytest -q`).
* No skipped tests in this phase (we’re testing “in isolation”; we’ll mark slow ones in later phases).
* Timing prints not required here (reserved for integration phases), but they can be used ad hoc during debugging.

---

## Notes & gotchas

* **Weights in `SequentialPCAUpdater`**: if your local branch hasn’t wired weights into the updater, U4 tests use the **representation**’s own `update_from_points(weights=...)`, not the updater. That keeps Phase 1 robust to current code state.
* Keep sample sizes modest (n≈200–400) to avoid flakiness and keep tests fast.
* Tolerances: use `0.95` for alignment on clean synthetic lines/planes; if local noise is larger, tune to `0.9`.

This plan gives you high-confidence, low-effort checks on the core mechanics before we move to end-to-end behavior.

===============================================================================

Here’s a concrete, file-by-file implementation plan for **Phase 2 — First integration pass**. No code yet—just what we’ll create and what goes in each file.

---

# Phase 2 — Implementation Plan

## 0) Update an existing helper (enable data for Phase 2)

### `tests/data_gen.py`

**Change:** implement the body of `make_two_lines_3d(...)`.

* **Function:** `make_two_lines_3d(n_per=200, noise=0.1, seed=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

  * **Behavior:**

    * Build two 3D line clusters:

      * Cluster 0 ≈ along `e_x` with small isotropic noise.
      * Cluster 1 ≈ along `e_y` with small isotropic noise.
    * Return:

      * `X`: shape `(2*n_per, 3)` (float32).
      * `y`: shape `(2*n_per,)`, labels `[0]*n_per + [1]*n_per`.
      * `G`: shape `(2, 3)`, ground-truth unit directions `[[1,0,0],[0,1,0]]`.
  * **Notes:** deterministic via `seed`; noise as provided.

*(The other generators remain `raise NotImplementedError` until later phases.)*

---

## 1) New test module for integration: I1 / I5 / I6

### `tests/integration/test_kfactors_i1_two_lines.py`

#### Imports

* `numpy as np`, `pytest`
* `torch` (optional—only to check device if needed)
* From project:

  * `from kfactors.algorithms import KFactors`
  * `from tests.utils import time_block, perm_invariant_axis_match, perm_invariant_accuracy`
  * `from tests.data_gen import make_two_lines_3d`

#### Test 1 — I1: Two 1D lines in 3D (happy path)

**Name:** `test_i1_two_lines_alignment_and_accuracy(seed_all, torch_device)`

* **Setup:**

  * Generate `(X, y_true, G) = make_two_lines_3d(n_per=200, noise=0.05, seed=7)`.
  * Instantiate `KFactors(n_clusters=2, n_components=1, random_state=0, device=torch_device)`.
* **Action:**

  * Wrap `model.fit(X)` in `with time_block("I1-two-lines", meta={"n": 400, "d": 3, "K": 2, "R": 1}):`
* **Assertions:**

  * **Basis alignment (permutation/sign invariant):**

    * Extract learned basis vectors into `(2, 3)` array `B`, each column unit-normalized.
    * `best_score, best_perm = perm_invariant_axis_match(B, G)`.
    * Require `best_score / 2.0 >= 0.9` (average |cos| ≥ 0.9).
  * **Clustering accuracy (label permutation invariant):**

    * Use `y_pred = model.labels_.cpu().numpy()` (or `.numpy()` if already np).
    * `acc = perm_invariant_accuracy(y_pred, split_index=len(X)//2)`.
    * Require `acc >= 0.9`.
  * **Shapes sanity:** `model.cluster_bases_` exists and has shape `(K, R, d)` (or equivalent internal layout we’re using), and vectors have non-zero norm.

#### Test 2 — I5: Label permutation & sign robustness

**Name:** `test_i5_permutation_and_sign_robustness(seed_all, torch_device)`

* **Setup:** Same generator as I1 but **row-permuted** dataset:

  * Build `X_perm = np.vstack([X[n_per:], X[:n_per]])`
  * `y_true_perm = np.hstack([y_true[n_per:], y_true[:n_per]])` (still 0/1 in order after permutation)
* **Action:** Fit `KFactors` with identical hyperparameters, wrapped in `time_block("I5-two-lines-permuted", meta=...)`.
* **Assertions:** Re-use **exact** alignment and accuracy checks from I1.
  *(This test ensures sign/permutation invariances in our assessors, and robustness of `fit` to input order.)*

#### Test 3 — I6: Objective trace sanity

**Name:** `test_i6_objective_trace_monotone_across_stages(seed_all, torch_device)`

* **Setup:** Use the I1 dataset (no permutation).
* **Action:** Fit `KFactors` wrapped in `time_block("I6-objective-trace", meta=...)`.
* **Assertions:**

  * Ensure `model.stage_history_` exists and is a list with length `R`:

    * `len(model.stage_history_) == model.n_components`.
  * Extract per-stage objectives: `objs = [s['objective'] for s in model.stage_history_]`.

    * **Monotonic (non-increasing) across stages**:
      For all `t>0`, `objs[t] <= objs[t-1] + eps` where `eps = 1e-8 * max(1.0, abs(objs[t-1]))`.
  * (Optional) Ensure reasonable iteration counts per stage: `1 <= s['iterations'] <= model.max_iter`.
  * Print iteration totals via `time_block` meta or an extra print if helpful (no hard assert on values).

---

## 2) Optional tiny helper for readability (no new file)

Inside `tests/integration/test_kfactors_i1_two_lines.py`, you may add a **private** helper:

* `_learned_basis_as_rows(model) -> np.ndarray`
  Extracts the `(K, R, d)` structure into a `(K, d)` matrix by taking the `[:, stage]` direction (for `R=1`, just the first column), normalizing to unit length, and moving to `cpu().numpy()`.

*(This keeps assertions clean and implementation-independent.)*

---

## 3) Skips & guards

* Add `pytest.mark.skipif(KFactors is None, reason="KFactors not importable")`.
* Tests rely on `seed_all`, `torch_device`, and `time_block` from Phase 0’s `conftest.py` / `utils.py`—no additional fixtures needed.

---

## 4) Success criteria (“gate to proceed”)

* All three tests pass locally on CPU:

  * **I1:** alignment ≥ 0.9 average |cos|, accuracy ≥ 0.9.
  * **I5:** same thresholds as I1, despite input permutation.
  * **I6:** per-stage objective non-increasing; timing printed.
* The timing output (from `time_block`) appears alongside cluster meta (n/d/K/R). No strict time assert—just visibility.

---

## 5) What we’ll *not* do in Phase 2

* No multi-stage `R>1` integration yet (will come later).
* No large-n/d performance benchmarks (Phase 4/5).
* No deep probabilistic checks on PPCA likelihoods (Phase 3+).

---

If this looks good, I’ll draft the test file stubs next and we can fill in `make_two_lines_3d` before running the first end-to-end green suite.

===============================================================================

Here’s a crisp, buildable plan for Phase 3.

# Phase 3 — Implementation Plan (files, tests, functions)

## Overview

Broaden coverage to validate **local basis learning** and **sequential extraction**:

* I2: two anisotropic Gaussians in higher-D (d=10), K=2, R=1
* I3: two planes in 3D, K=2, R=2

We’ll implement the missing data generators, then add two integration test files. We’ll also add tiny test-local helpers (or, if you prefer, we can promote them later into `tests/utils.py`).

---

## 1) Update generators

### File: `tests/data_gen.py`

**Implement the two stubs from Phase 0:**

1. `make_aniso_gaussians(...) -> (X, y, (GA, GB))`

* **Inputs:** `n_per, d, covA, covB, rotA=None, rotB=None, seed=None`
* **Behavior:**

  * Construct two Gaussian clusters in ℝᵈ:

    * Σ\_A = (rotA @ covA @ rotAᵀ) if rotA else covA
    * Σ\_B = (rotB @ covB @ rotBᵀ) if rotB else covB
    * μ\_A = 0, μ\_B = 0 (keep it simple/centered).
  * Sample `n_per` from each.
  * Return:

    * `X`: (2·n\_per, d) float32
    * `y`: (2·n\_per,) int64 (first half 0, second half 1)
    * `(GA, GB)`: ground-truth **top principal axes** (unit vectors) of Σ\_A and Σ\_B (columns as vectors). (Compute by eigendecomposition; choose the top-eigenvector.)
  * **Doc note:** keep variances strongly anisotropic (e.g., λ₁ ≫ λ₂ ≥ …).

2. `make_two_planes_3d(...) -> (X, y, (G1, G2))`

* **Inputs:** `n_per, noise, seed=None`
* **Behavior:**

  * Two clusters, each supported on a distinct 2D plane in ℝ³:

    * Choose two orthonormal bases `G1` and `G2` (2×3 each), e.g., one plane near span{e₁,e₂}, another near span{(e₁+e₃)/√2, (e₂−e₃)/√2}, but keep the planes distinct.
    * Sample coefficients `t ~ N(0,1)` in ℝ² for each cluster; map via the plane basis; add small isotropic noise in ℝ³ (`noise`).
  * Return:

    * `X`: (2·n\_per, 3), `y` as before,
    * `(G1, G2)`: the **plane bases** (each 2×3, rows or columns are unit vectors; document the convention).

**Order:** Do these first. They’re prerequisites for I2/I3.

---

## 2) New integration tests

### File: `tests/integration/test_kfactors_i2_aniso_gaussians.py`

**Purpose:** K=2, R=1 in d=10; learned direction per cluster ≈ top PC of that cluster.

**Tests:**

1. `test_i2_alignment_and_accuracy(seed_all, torch_device)`

* **Setup:**

  * Define `d=10`, `n_per≈400`, strongly anisotropic diagonals:

    * `covA = diag([5.0, 2.0, 1.0, …])`
    * `covB = diag([4.5, 2.5, 1.0, …])`
  * Optional random rotations `rotA`, `rotB` (orthonormal from QR of random Gaussian).
  * `X, y_true, (GA, GB) = make_aniso_gaussians(...)`.
  * `K=2, R=1`; instantiate `KFactors(..., random_state=seed, device=fixture)`.
* **Timing:** wrap `model.fit(X)` in `with time_block("I2", meta={"n":..., "d":..., "K":2, "R":1}):`
* **Alignment:**

  * Extract learned basis rows `B` using the `_learned_basis_as_rows(model, expected_dim=d)` helper pattern from Phase 2.
  * Build `G = stack([GA, GB])` (shape (2, d)).
  * Use `perm_invariant_axis_match(B, G)`; assert `avg |cos| ≥ 0.90`.
* **Accuracy:**

  * `y_pred = model.predict(X).cpu().numpy()`.
  * `acc = perm_invariant_accuracy(y_pred, split_index=X.shape[0]//2)`; assert `acc ≥ 0.85`.
* **Shape sanity:** accept `(K, d, R)` or `(K, R, d)` like in Phase 2.
* **Parametrization for flake check:** run over 3 seeds, e.g. `@pytest.mark.parametrize("seed", [3, 11, 47])` to ensure stability.

2. (Optional) `test_i2_objective_monotone_basic(...)`

* Mirror Phase 2’s objective non-increase sanity; keep threshold lax (epsilon).

---

### File: `tests/integration/test_kfactors_i3_two_planes_r2.py`

**Purpose:** K=2, R=2 in ℝ³; **sequential extraction** within cluster learns an in-plane orthonormal basis close to the ground-truth plane; stage-1 aligns with a top in-plane axis; stage-2 is in-plane and near-orthogonal to stage-1.

**Tests:**

1. `test_i3_subspace_alignment_and_orthogonality(seed_all, torch_device)`

* **Setup:**

  * `X, y_true, (G1, G2) = make_two_planes_3d(n_per≈400, noise≈0.05, seed=seed)`
  * Fit `KFactors(K=2, R=2, ...)` inside `time_block("I3", meta={"n":..., "d":3, "K":2, "R":2})`.
* **Extract learned per-cluster 2D bases:**

  * Build a helper in the test (local) to return `(K, 2, d)`:

    * Use `model.cluster_bases_`, handle `(K, d, R)` or `(K, R, d)`, and **normalize columns**, returning per-cluster rows or columns consistently.
* **Subspace alignment metric (2D vs 2D):**

  * For each cluster k, and each ground-truth plane basis Gk (2×3), compute **principal angles** between the learned 2D subspace Lk and Gk:

    * SVD of `Gkᵀ Lk` → singular values `σ₁, σ₂` = cosines of principal angles.
  * Accept if **both σ₁, σ₂ ≥ 0.90** (or equivalently, min singular ≥ 0.90).
  * Because clusters can swap, compute the best matching permutation `{0,1}` and assert both clusters pass under that permutation.
  * (If you’d rather not add a new helper to `utils` yet, define a small `principal_angles_2d(G, L)` inside this test file.)
* **Within-cluster orthogonality:**

  * For each learned cluster basis `Lk = [v¹, v²]`, assert `|⟨v¹,v²⟩| ≤ 0.15`.
* **Accuracy:** Same `perm_invariant_accuracy(...) ≥ 0.85`.
* **Parametrization:** 3 seeds (e.g. `[5, 17, 101]`) to check non-flakiness.

---

## 3) (Optional) tiny test-local helpers

You can keep these helpers **inside** the new test files for Phase 3. Later, if we like them, we can promote to `tests/utils.py`.

* `_learned_basis_matrix(model, expected_dim: int, R: int) -> np.ndarray`

  * Returns `(K, R, d)` unit vectors, handling `(K, d, R)` or `(K, R, d)`, with robust normalization.

* `principal_angles_2d(G: np.ndarray, L: np.ndarray) -> tuple[float, float]`

  * Returns the singular values of `Gᵀ L` (cosines). We’ll check both ≥ threshold.

---

## 4) Order of work

1. **Implement generators** in `tests/data_gen.py`:

   * `make_aniso_gaussians` (with deterministic cov/rot; verify shapes).
   * `make_two_planes_3d` (two distinct planes; verify orthonormal bases).
2. **Write I2 test** single seed → tune thresholds if needed → parametrize 3 seeds.
3. **Write I3 test** single seed → tune orthogonality & subspace thresholds → parametrize 3 seeds.
4. Ensure timing prints appear via `time_block` in both tests.
5. Run full suite × a couple of times to observe stability.

**Gate to proceed:** Both I2 and I3 green across 3 seeds, with stable timings (no significant flakiness).

---

## 5) Notes & thresholds

* **Thresholds:** Start at 0.90 for alignment / subspace overlap and 0.85 for accuracy (as specified). If we see rare fail at a single seed boundary, consider 0.88/0.83, but only if justified by debug output.
* **Shapes:** Keep the axis-agnostic stance for `cluster_bases_` (we already set this precedent in Phase 2).
* **Noisy planes:** If `noise` is too high, lower it slightly before relaxing thresholds; we want meaningful signal, not brittle tests.

That’s it. When you’re ready, I’ll draft the data generators and then the I2/I3 tests.

===============================================================================


Here’s a concrete implementation plan for **Phase 4 — Penalty behavior as seen end-to-end**. It mirrors Phase 0–3 structure: one new test module, tiny synthetic setups, and assertions that target exactly the “eats credit” heuristic.

---

# Phase 4 — Implementation Plan

## New file

### `tests/integration/test_kfactors_i4_penalty_behavior.py`

**Purpose:** Verify, in a minimal but end-to-end way (assignment strategy + direction tracking), that previously claimed directions *reduce* effective weight for aligned candidates and *do not* reduce it for orthogonal candidates — and that the numeric values match the closed-form heuristic within a tight tolerance.

We’ll use the actual `PenalizedAssignment` and `DirectionTracker` working together (not just unit computations), and we’ll compute “stage weights” in the same way `KFactors.fit` does (i.e., `gather` the per-point penalty for the assigned cluster). We’ll time each test with `time_block`.

> Note: We don’t need to run a full `KFactors.fit` here (which doesn’t currently persist per-stage weights in history). Instead, we’ll mimic a 2-stage flow: (a) create a claimed direction history (stage 0), then (b) run `PenalizedAssignment.compute_assignments` for a “stage 1” candidate set and read `aux_info['penalties']` directly. This uses the same code paths as the algorithm’s assignment step.

---

## Tests to add

### 1) `test_i4_penalty_aligned_vs_orthogonal_end_to_end_product()`

**Goal:** Show that with the **product** penalty, an aligned candidate gets smaller penalty (≈0.0) than an orthogonal candidate (≈1.0).

**Setup:**

* Dimension `d=2`, `K=2`, `n=1`.
* Orthobasis `e0=[1,0]`, `e1=[0,1]`.
* Create a `DirectionTracker(n_points=1, n_clusters=2, device=cpu)` and record a prior claim for the point: `(e0, weight=1.0)` via `add_claimed_direction`.
* Construct two simple PPCA (or Subspace) representations for the clusters with **current** basis directions:

  * Cluster 0: `W[:,0]=e0` (aligned with claim).
  * Cluster 1: `W[:,0]=e1` (orthogonal to claim).
* Call `PenalizedAssignment(penalty_type='product', penalty_weight=1.0).compute_assignments(points, representations, direction_tracker=tracker, current_stage=0)`.
* Extract `aux_info['penalties']` → shape `(n, K)`.

**Assertions:**

* `penalty_aligned ≈ 0.0 ± 0.02`, `penalty_orthogonal ≈ 1.0 ± 0.02`.
* Print timing with `time_block("I4-product", meta={"n":1,"K":2,"R":1})`.

---

### 2) `test_i4_penalty_numeric_match_weighted_product()`

**Goal:** Verify the **weighted** product heuristic matches the closed form `∏(1 − a_j |cos θ_j|)` within ±0.02.

**Setup:**

* Same geometry as above, but record a **partial** prior claim: `(e0, weight=0.7)`.
* Same candidate directions: `e0` (aligned), `e1` (orthogonal).
* Compute penalties via `compute_assignments(...)` and read `aux_info['penalties']`.

**Expected values:**

* Aligned: `1 − 0.7*1 = 0.3` → expect `≈ 0.30 ± 0.02`.
* Orthogonal: `1 − 0.7*0 = 1.0` → expect `≈ 1.00 ± 0.02`.

**Assertions:**

* Numeric closeness to the two targets above.
* Print timing with `time_block("I4-weighted-product", meta={...})`.

---

### 3) `test_i4_penalty_mean_mode_numeric()`

**Goal:** Verify the **mean** (a.k.a. “sum”) penalty mode `1 − (∑ a_j |cos θ_j| / ∑ a_j)` matches expectation.

**Setup:**

* Same as test (2) but instantiate `PenalizedAssignment(penalty_type='sum', ...)`.
* Partial prior claim `(e0, weight=0.7)`.

**Expected values:**

* Aligned: `1 − (0.7*1 / 0.7) = 0.0` → expect `≈ 0.00 ± 0.02`.
* Orthogonal: `1 − (0.7*0 / 0.7) = 1.0` → expect `≈ 1.00 ± 0.02`.

**Assertions & timing** as before.

---

### 4) `test_i4_effective_sample_sizes_log_product()`

**Goal:** Demonstrate how effective per-cluster sample sizes (the sums of stage weights the algorithm would feed to updates) reflect the penalty. This mirrors what `KFactors.fit` does: `stage_weights = penalties.gather(1, assignments[:,None]).squeeze(1)`.

**Setup:**

* Tiny batch: `n=4` points at simple coordinates; prior claims:

  * Points 0–1: `(e0, weight=1.0)`.
  * Points 2–3: `(e0, weight=0.7)`.
* Candidate directions per cluster: cluster 0 uses `e0` (aligned); cluster 1 uses `e1` (orthogonal).
* Compute `assignments, aux_info = compute_assignments(...)`.
* Build `stage_weights = aux_info['penalties'].gather(1, assignments[:,None]).squeeze(1)`.
* For bookkeeping, compute

  * `eff_n_k = sum(stage_weights[p] for p assigned to cluster k)`, for `k=0,1`.

**Assertions:**

* `eff_n_0` (aligned) is **smaller** than `eff_n_1` (orthogonal), and numerically consistent with the closed form:

  * For points with weight 1.0: aligned = 0.0, orthogonal = 1.0
  * For points with weight 0.7: aligned ≈ 0.3 (product), orthogonal = 1.0
  * So `eff_n_0 ≈ 0.0 + 0.3` (from the 0.7-weight pair), `eff_n_1 ≈ 2.0 + 2.0 = 4.0` if all 4 assigned to their respective argmins (or similar if ties—allow a small tolerance).
* Print a summary line via `time_block("I4-effective-n", meta={"eff_n": {0: eff_n_0, 1: eff_n_1}})`.

> If you prefer to avoid argmin dependence, set the **distances equal** for both clusters (e.g., center points at the origin so residuals are equal), then force a deterministic assignment (e.g., pick cluster via index parity) to make the `gather` deterministic. The heuristic under test is the *penalty*, not cluster choice.

---

## Notes / Guardrails

* Keep everything on CPU for reproducibility.
* Use `np.allclose(..., atol=0.02)` for numeric checks.
* Reuse the same canonical directions (`e0`, `e1`) and tiny datasets so debugging is trivial.
* We don’t modify library code; tests operate through public classes:

  * `DirectionTracker` to inject claimed history (+ weights),
  * `PenalizedAssignment.compute_assignments(...)` to compute penalties and assignments,
  * and then reproduce the stage-weight gather exactly as `KFactors.fit` does.

---

## Exit criteria (Gate)

* All I4 tests pass consistently across a few seeds (though these tests are mostly deterministic).
* Console prints from `time_block` include the timing lines and effective sample size dicts that “look right” at a glance (e.g., aligned cluster’s `eff_n` far smaller than orthogonal’s under product mode).

Once this is green, we’ll have good empirical evidence that the penalty heuristic **does** “eat credit” as intended when seen through the same API surfaces K-Factors uses during training.


===============================================================================

Awesome—here’s a tight, copy-pasteable implementation plan for **Phase 5 — Determinism & edges**. No code yet; just what to create and what each test will do.

---

# Phase 5 — Implementation plan

## New / updated files

### 1) `tests/integration/test_kfactors_i7_determinism.py`

**Purpose:** with fixed seed & same device, two runs produce the same **labels** (up to permutation) and the same **bases** (up to permutation/sign) within `1e-6`.

**Tests**

* `test_i7_two_runs_same_results_on_two_lines()`

  * **Data:** reuse `make_two_lines_3d(n_per=300, noise=0.05, seed=...)`.
  * **Config:** `K=2, R=1`, `random_state=SEED`, CPU device.
  * **Run:** fit `KFactors` twice with identical inputs/settings; wrap each fit with `time_block("I7-two-lines", meta={...})`.
  * **Assertions:**

    * **Labels:** use `perm_invariant_accuracy(y_pred, split_index=n_per)` on each run against the *same* ground truth; both runs should achieve identical accuracy; additionally, compute the **best label permutation** between run1 and run2 predictions and assert equality after remapping.
    * **Bases:** extract `model.cluster_bases_` from both runs; normalize; align **clusters** via best permutation on their first basis vector (R=1 is simple); for each matched pair, align **sign** and assert `np.allclose` with `atol=1e-6`.
* `test_i7_two_runs_same_results_on_aniso_gaussians()`

  * **Data:** reuse `make_aniso_gaussians(n_per=300, d=10, covA, covB, rotA, rotB, seed=...)` (same settings as Phase 3).
  * **Config:** `K=2, R=1`, `random_state=SEED`, CPU.
  * **Run/Asserts:** identical pattern as above (labels up to permutation; bases up to sign/permutation within `1e-6`).
  * **Timing:** `time_block("I7-aniso", meta={...})`.

> Notes:
>
> * Keep everything on **CPU** (use your `torch_device` fixture) to avoid CUDA nondeterminism.
> * Be explicit that “equal after sign/permutation” is the criterion—PCA directions are defined up to sign and order.

---

### 2) `tests/integration/test_kfactors_i8_edges.py`

**Purpose:** stress small/degenerate shapes; ensure **no exceptions**, objective finite, and we get timing.

**Tests (parametrized or separate):**

* `test_i8_small_n_3d_two_lines()`

  * **Data:** `make_two_lines_3d(n_per=3, noise=0.05, seed=...)` → only 6 points.
  * **Config:** `K=2, R=1`, moderate `max_iter` (e.g., 30).
  * **Run:** `time_block("I8-small-n", meta={n=6,d=3,K=2,R=1})`; `model.fit(X)`.
  * **Assertions:** no exception; `np.isfinite(model.inertia_)`; `len(model.cluster_bases_) == K`; history exists (`len(model.history_) >= 1`).
* `test_i8_high_dim_low_n()`

  * **Data:** synthesize `n=100, d=200` with two or three blobs (e.g., random means, small isotropic noise); labels not used for accuracy—this is a stability test.
  * **Config:** `K=3, R=2`, `max_iter` modest; CPU.
  * **Run:** `time_block("I8-hdln", meta={n,d,K,R})`.
  * **Assertions:** fit completes; `inertia_` finite; `cluster_bases_` has shape `(K, ?, ?)`, with each basis column having non-zero norm (e.g., ≥1e-9).
* `test_i8_colinear_points()`

  * **Data:** all points on a single line in 2D (e.g., `t*[1,0] + small_noise`), maybe 200 points.
  * **Config:** try `K=2, R=1` (intentionally over-cluster); CPU.
  * **Run:** `time_block("I8-colinear", meta={...})`.
  * **Assertions:** no exception; finite objective; the two learned directions both have non-zero norm; (optional) their **|dot|** may be close to 1—fine for this edge.
* `test_i8_zero_variance_feature()`

  * **Data:** take `make_two_lines_3d(...).X` and append a constant column → `d=4` with one zero-variance feature.
  * **Config:** `K=2, R=1`.
  * **Run:** `time_block("I8-zero-var", meta={...})`.
  * **Assertions:** no exception; finite objective; the added constant dimension’s coordinate in basis vectors is numerically small (optional sanity check, e.g., `abs(basis_component) <= 1e-3` median).

> These four cover: tiny sample size, high-d/low-n regime, degenerate geometry, and degenerate feature variance. No accuracy thresholds—just stability and finiteness.

---

### (Optional) tiny helper additions (only if you want them)

You can keep all alignment logic inline in the tests, or add two small helpers to `tests/utils.py`:

* `best_cluster_permutation(basesA: np.ndarray, basesB: np.ndarray) -> tuple[perm, scores]`

  * For `R=1`, uses a simple K×K matrix of `|cos|` between cluster axes and returns argmax permutation (Hungarian or greedy).
* `align_sign(u: np.ndarray, v: np.ndarray) -> np.ndarray`

  * Returns `v` or `-v` to maximize dot with `u`.

These are quality-of-life only; not required if you keep R=1 in I7.

---

## Exit gate

* **I7**: both determinism tests green; label parity after permutation; bases match within `1e-6` after sign/permutation; timing printed.
* **I8**: all edge tests complete without exceptions; each prints duration; `inertia_` finite in each case.

Once these pass, you’ll have locked down reproducibility and robustness against the usual gremlins—exactly the guardrails you want before iterating on heuristics or performance.


===============================================================================

