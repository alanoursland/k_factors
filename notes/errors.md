
**Weighted updates get overwritten**

   * In `SequentialPCAUpdater.update`, you do a weighted centering path, then immediately recompute an unweighted mean and overwrite `centered`. That nullifies weighting.
   * Remove the second mean/centering block so the weighted path actually flows into the SVD/eigh.

**`to(device)` for representations**

   * `BaseRepresentation.to` re-instantiates via `self.__class__(self._dimension, device)`. Subclasses with extra ctor args (e.g. `SubspaceRepresentation(d, r, device)`) will break.
   * You already override `PPCARepresentation.to` (great). Do the same for `SubspaceRepresentation`, or refactor the base `to()` to use `__reduce__/state` style cloning rather than calling the subclass constructor.

**Double initialization in base `_fit`**

   * `BaseClusteringAlgorithm._fit` calls `initialization_strategy.initialize(...)` and then `_create_representations(...)` which may (re)initialize again. For algorithms that *don’t* override `fit`, that first init can be wasted or confusing.
   * Easiest: pass the initializer’s outputs into `_create_representations` or let `_create_representations` own initialization entirely.

**Metric shift inside PPCA assignments**

   * During early stages, PPCA distances are Euclidean residuals to a *partial* basis; after exhaustion they switch to PPCA Mahalanobis. That’s a valid design choice—just be aware the scale/units change across branches (it can affect the objective trend you log).

### small nits / polish

* In `KFactorsObjective.compute`, you ignore penalties (by design), but if you adopt penalized selection you may want a companion *reporting* metric that includes the penalty term for monitoring.

* `DirectionTracker` is clean; if you expect very large `n`, consider an optional compact history (e.g., cap per-point entries to `R`) or pack weights/dirs into tensors for batched adds (you already do batched penalties, which is great).

* `inverse_transform`’s assignment inference picks the first nonzero block—fine for the demo; just flagging it’s not permutation-robust if multiple blocks are populated.

