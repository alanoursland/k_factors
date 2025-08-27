# SequentialPCAUpdater — Design Guide

> Update strategy that extracts **one principal direction per stage** from a cluster’s assigned points. It supports both PPCA-like and plain subspace representations and keeps columns orthogonal as stages progress.

---

## Role in K-Factors

Within each stage $t = 0,1,\dots$, K-Factors alternates:

1. **Assignment** (with directional penalty),
2. **Update** of each cluster’s representation for the **current stage**.

`SequentialPCAUpdater` implements step (2): given the points currently assigned to a cluster, it updates:

* the cluster **mean** $\mu_k$,
* the **new basis column** $w^{(t)}_k$ (or `basis[:, t]`),
* and, for PPCA, the **noise variance** after the final stage.

---

## API & Call Contract

```python
class SequentialPCAUpdater(ParameterUpdater):
    def __init__(self, current_stage: int = 0)

    def update(
        self,
        representation: ClusterRepresentation,   # PPCARepresentation | SubspaceRepresentation
        points: Tensor,                          # (n_k, d) assigned points
        assignments: Optional[Tensor] = None,    # ignored
        current_stage: Optional[int] = None,
        **kwargs
    ) -> None
```

* **Stage source:** if `current_stage` is passed to `update`, it overrides the instance’s `self.current_stage`.
* **Empty cluster:** if `len(points) == 0`, the call is a no-op.
* **Mean update:** the cluster **mean is always recomputed** from assigned points:
  $\mu_k \leftarrow \text{mean}(X_k)$; then points are **centered**.

---

## Core algorithm (per cluster, per stage)

### 1) Residualization by previously learned columns

* If `stage == 0`: residuals $R = X - \mu$.
* If `stage > 0`: project out previously learned columns to isolate the next direction:

  $$
  R = (X - \mu) - (X - \mu)\,B_{1:t-1}\,B_{1:t-1}^\top,
  $$

  where $B$ is `W` (PPCA) or `basis` (Subspace) and `B_{1:t-1}` are the first `stage` columns.

### 2) Extract the top direction of the residuals

* **PPCARepresentation**

  * Computes $\mathrm{cov} = \frac{1}{n} R^\top R$.
  * Tries `torch.linalg.eigh(cov)` to get top eigenpair; on failure, falls back to `safe_svd(R)`.
  * Enforces **sign consistency** by flipping the eigenvector if its first entry is negative.
  * Initializes `_W` on the first stage; writes the `stage` column as:

    $$
    W[:, \text{stage}] \leftarrow \sqrt{\max(\lambda_{\text{top}}, 10^{-6})}\,\;u_{\text{top}},
    $$

    then calls `orthogonalize_basis(W[:, :stage+1])` to keep columns orthonormal (numerically).

* **SubspaceRepresentation**

  * Prefers SVD on residuals: `U, S, Vt = safe_svd(R, full_matrices=False)`; `top_direction = Vt[0]`.
  * (Fallback: covariance + `eigh` if SVD fails.)
  * Enforces sign consistency as above.
  * Initializes `_basis` on the first stage and writes `basis[:, stage] = top_direction`, then orthogonalizes `basis[:, :stage+1]`.

### 3) Noise variance for PPCA (at the last stage only)

When `stage == latent_dim - 1`:

* Recompute residuals against the **final basis** $W$:
  $R_\text{final} = (X - \mu) - (X - \mu) W W^\top$.
* Set isotropic variance:

  $$
  \sigma^2 \leftarrow \frac{1}{d}\,\mathrm{mean}\big(\lVert R_\text{final}(i)\rVert_2^2\big).
  $$
* Mark caches dirty: `representation._needs_cache_update = True`.

---

## Representations & shapes

* **PPCARepresentation**

  * Assumes attributes: `mean: (d,)`, `latent_dim: int`, `W: (d, r)` (property), backing storage `_W: (d, r)`, `variance: ()`, and `_needs_cache_update: bool`.
  * Columns of `W` are maintained **orthonormal** (after scaling/orthogonalization).

* **SubspaceRepresentation**

  * Assumes attributes: `mean: (d,)`, `subspace_dim: int`, `basis: (d, r)` (property), backing storage `_basis: (d, r)`.
  * Columns of `basis` are maintained **orthonormal**.

* **Orthogonalization helpers**

  * `safe_svd` and `orthogonalize_basis` are delegated to `[utils/linalg.py]` (broken link).
    They provide numerical stability and ensure orthonormal columns after each stage write.

---

## Numerical choices & stability

* **Eigen vs SVD**:

  * PPCA path uses **eigendecomposition of residual covariance** (fast when $d$ is modest) and falls back to SVD.
  * Subspace path prefers **SVD of residuals** (often more stable on rank-deficient data), with covariance fallback.
* **Sign convention**: flipping direction if its first component is negative makes stage updates deterministic given a seed.
* **Clamping eigenvalue**: $\sqrt{\max(\lambda, 10^{-6})}$ avoids zeroing out columns if `eigh` returns a tiny/negative top eigenvalue due to numerical errors.

---

## Complexity (per cluster, per stage)

Let $n_k$ be points in the cluster, $d$ the ambient dimension.

* **Residualization**: $O(n_k d r)$ to project out `r = stage` columns.
* **PPCA (covariance + eig)**: build covariance $O(n_k d^2)$; eigendecomposition $O(d^3)$.
* **Subspace (SVD)**: $O(n_k d \min(n_k, d))$.
* **Orthogonalize**: $O(d r^2)$ (thin QR-like cost on a $d \times (r+1)$ block).

---

## Behavior by stage & edge cases

* **Initialization per stage 0**: `_W` or `_basis` zero-initialized with correct shape; subsequent stages assume the backing storage exists.
* **Stage bounds**: Writes only if `stage < latent_dim` (PPCA) or `stage < subspace_dim` (Subspace). If callers request a larger stage, the update safely skips the write.
* **Empty clusters**: Update exits early; mean/basis are unchanged.
* **Assignments argument**: accepted for `ParameterUpdater` compatibility but **ignored** here; updates are driven purely by the points supplied.

---

## Design intents (not requirements)

* **Sequential extraction**: each new column is learned from **residuals** after removing previously extracted columns, ensuring directions are orthogonal and ordered by explained residual variance.
* **PPCA scaling**: columns of `W` are scaled by $\sqrt{\lambda_\text{top}}$. Final isotropic noise is set after the last column based on remaining residual energy.
* **Subspace simplicity**: stores **unit** directions; no variance parameter.

---

## Integration touch-points

* Called from K-Factors’ inner loop with `current_stage=stage`.
* Pairs with `PenalizedAssignment`, which computes distances using **partial bases** up to `stage`.
* After a stage converges, K-Factors uses the updated columns to **claim** directions in the `DirectionTracker` for penalty computation in the next stage.

---

## Reference (related components)

* Representations:

  * `[representations/ppca.py]` *(broken link)*
  * `[representations/subspace.py]` *(broken link)*
* Linear algebra utilities:

  * `[utils/linalg.py]` *(broken link)*
* K-Factors algorithm orchestration:

  * `[algorithms/k_factors.py]` *(broken link)*

---

*This updater is the “M-step for one column”: it maintains orthonormal bases, works on residuals, and sets PPCA’s noise variance at the final stage for a clean separation between basis learning and variance estimation.*
