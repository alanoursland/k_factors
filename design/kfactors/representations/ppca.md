# PPCARepresentation — Design Guide

> A probabilistic PCA (PPCA) cluster representation used by K-Factors/C-Factors. Each cluster is modeled as a Gaussian with **low-rank** covariance:
>
> $$
> x \sim \mathcal{N}\!\big(\mu,\; C\big),\qquad C = W W^\top + \sigma^2 I_d
> $$
>
> where $W\in\mathbb{R}^{d\times r}$ (with $r<d$) and $\sigma^2>0$ is isotropic noise.

---

## Purpose & Role

* Provides the **cluster-local generative model** used for distances, log-likelihood, sampling, and posterior inference over latents.
* Integrates with **SequentialPCAUpdater** (stagewise updates of $W$) and with generic EM/ML updates via `update_from_points`.
* Supplies parameters to the framework via `get_parameters`/`set_parameters`.

---

## Construction & State

```python
PPCARepresentation(
    dimension: int,           # d
    latent_dim: int,          # r, must be < d
    device: torch.device,
    init_variance: float = 1.0
)
```

* Validates `latent_dim < dimension`.
* Initializes:

  * `_W ∈ ℝ^{d×r}` with **random orthonormal columns** via QR; scaled by `sqrt(init_variance)`.
  * `_variance = σ²` (clamped to ≥ 1e-6 on writes).
  * `_M_inv = (I_r + W^T W / σ²)^{-1}` cache (computed lazily).
  * `_needs_cache_update = True`.
* Inherits from `BaseRepresentation` (not shown) which provides `_mean ∈ ℝ^d`, `_dimension`, `_device`, shape checks, etc.

### Properties

* `W: (d, r)` getter/setter (setter marks cache dirty).
* `variance: ()` getter/setter (setter clamps ≥ 1e-6 and marks cache dirty).

---

## Cached Linear Algebra

To compute distances and posteriors efficiently, it uses the **Woodbury identities**. Cache update:

$$
M \;=\; I_r + \frac{1}{\sigma^2} W^\top W,\qquad
M^{-1} \text{ cached, via inv or pinv}.
$$

`_update_cache()` recomputes $M^{-1}$ when `W` or `σ²` change. For `r=0`, `_M_inv=None`.

---

## Distances (Mahalanobis under PPCA)

```python
distance_to_point(points: Tensor[n, d]) -> Tensor[n]
```

Computes $(x-\mu)^\top C^{-1} (x-\mu)$ using:

$$
C^{-1} \;=\; \frac{1}{\sigma^2}I_d \;-\; \frac{1}{\sigma^2}\,W\,M^{-1}W^\top\,\frac{1}{\sigma^2}.
$$

Implementation outline:

1. Center: $X_c = X - \mu$.
2. `term1 = ||X_c||^2 / σ²`.
3. `Wt_centered = X_c W / sqrt(σ²)`.
4. `M_inv_Wt_centered = Wt_centered M^{-1}`.
5. `term2 = ⟨Wt_centered, M_inv_Wt_centered⟩ / σ²` (rowwise).
6. `distance = term1 - term2`.

Special case `r=0`: `||X_c||^2 / σ²`.

**Notes**

* Returns **squared Mahalanobis** distances (not Euclidean residuals).
* Batch-vectorized and device-aware.

---

## Log-Likelihood

```python
log_likelihood(points: Tensor[n, d]) -> Tensor[n]
```

$$
\log |C| = d\log\sigma^2 + \log|I_r + W^\top W / \sigma^2|
$$

$$
\log p(x) = -\tfrac12\Big[\,(x-\mu)^\top C^{-1}(x-\mu) + \log|C| + d\log(2\pi)\Big].
$$

Uses `slogdet` on $M$ for stability.

---

## Latent Posterior $p(z|x)$

```python
posterior_mean_cov(point: Tensor[d]) -> (Tensor[r], Tensor[r, r])
```

Closed-form PPCA posterior:

$$
\Sigma_{z|x} = M^{-1},\qquad
\mu_{z|x} = M^{-1} \frac{1}{\sigma^2} W^\top (x-\mu).
$$

For `r=0`, returns empty tensors.

---

## Sampling / Generation

* `sample_latent(n) → 𝒩(0,I_r)` latent draws $z$.
* `generate_samples(n)`:

  $$
  x = \mu + W z + \epsilon,\qquad \epsilon \sim \mathcal{N}(0, \sigma^2 I_d).
  $$

---

## Parameter Updates (non-sequential)

```python
update_from_points(points, weights=None)
```

* Updates **mean** (weighted if `weights` provided).
* Forms **weighted** centered data when soft assignments.
* If `r=0`: sets `σ² = mean(||X_c||^2)/(n·d)` and returns.
* Else computes sample covariance $S = X_c^\top X_c / n$ and performs eigendecomposition:

  * Sort eigenpairs descending.
  * If `r < d`:

    * $U_r$: top `r` eigenvectors; `λ_r`: corresponding eigenvalues.
    * $\sigma^2 \leftarrow \text{mean of remaining eigenvalues}$ (or smallest eigen if edge case).
    * $W \leftarrow U_r \operatorname{diag}\!\big(\sqrt{\max(λ_r - \sigma^2, 10^{-6})}\big)$.
  * Else (degenerate): `W = U[:, :r]`, `σ² = 1e-6`.
* Marks cache dirty.

**This is the classic PPCA closed-form ML update (Tipping & Bishop, 1999).**

---

## Interaction with Sequential Updates

* **SequentialPCAUpdater** writes columns of $W$ **stagewise** and orthogonalizes them, then (at final stage) sets `variance` via residual energy.
* Both pathways (batch ML vs. sequential) are compatible; both ensure cache consistency.

---

## Shapes, Devices, Invariants

* `W.shape == (d, r)`, `variance.shape == ()`, `mean.shape == (d,)`.
* `latent_dim = r`, `dimension = d`, `r < d`.
* All tensors reside on `_device`; `.to(device)` constructs a **new** `PPCARepresentation` and copies parameters.
* `variance` clamped to ≥ `1e-6`. Distance/posterior require `_update_cache()` to have run (handled lazily).
* Sign conventions for eigenvectors are not enforced here (sequential updater does deterministic sign handling).

---

## Numerical & Performance Notes

* Uses Woodbury structure → distance is **O(ndr + nr^2 + n)** after caching (no $d\times d$ solves).
* Cache inversion of $M\in\mathbb{R}^{r\times r}$ is cheap (small `r`). Falls back to `pinv` if needed.
* For ill-conditioned $S$, eigen fallback warnings are surfaced in `update_from_points`.

---

## Public API Summary

* Parameters:

  * `get_parameters()` → `{ 'mean': (d,), 'W': (d, r), 'variance': () }` (cloned).
  * `set_parameters(dict)` → sets any subset; clamps variance; invalidates cache.
* Computations:

  * `distance_to_point(X)` → squared Mahalanobis under PPCA.
  * `log_likelihood(X)` → per-point log pdf.
  * `posterior_mean_cov(x)` → posterior over `z`.
  * `sample_latent(n)`, `generate_samples(n)`.
  * `update_from_points(X, weights=None)` → ML/EM-style batch update.
* Utilities:

  * `to(device)` → new moved instance.
  * `__repr__` human-readable summary.

---

## How It Fits Into K-Factors

* **Assignments:** `PenalizedAssignment` may use PPCA’s distances or **stage-aware residuals** based on the current basis columns.
* **Updates:** `SequentialPCAUpdater` modifies `W` one column at a time and sets `variance` at the final stage.
* **Transform/Inverse:** Other components may project using $W$ and reconstruct via $\mu + W z$.

---

*This PPCA module provides a principled, efficient Gaussian subspace model that plays nicely with both batch ML updates and K-Factors’ sequential, stage-driven learning.*
