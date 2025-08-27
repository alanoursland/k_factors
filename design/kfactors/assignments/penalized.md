# PenalizedAssignment — Design Guide (K-Factors)

> This strategy assigns each point to a cluster by minimizing a **penalized residual**. The penalty discourages a point from repeatedly aligning with the **same directions** it has already “claimed” in earlier stages, pushing it toward **diverse subspace usage**.

---

## Overview

Given:

* data `points ∈ ℝ^{n×d}`,
* cluster representations `reps = {R_k}` (PPCA or subspace-like),
* current stage index `t = current_stage`,
* a `DirectionTracker` with each point’s previously claimed directions,

the strategy computes:

1. a **stage-aware residual** $\mathrm{dist}_{ik}$ for each point $i$ vs. cluster $k$,
2. a **directional penalty scale** $p_{ik}$ from `DirectionTracker` based on how similar the candidate direction $u_k^{(t)}$ is to directions the point $i$ has already claimed,
3. a **penalized distance**:

$$
\boxed{\;\widetilde{\mathrm{dist}}_{ik} \;=\; \mathrm{dist}_{ik}\cdot\big((1-\alpha)+\alpha\,p_{ik}\big)\;}
$$

where $\alpha = \texttt{penalty\_weight} \in [0,1]$.

Assignments are **hard**:

$$
\widehat{k}(i) \;=\; \arg\min_{k} \;\widetilde{\mathrm{dist}}_{ik}.
$$

Returned `aux_info` includes raw distances, penalties, penalized distances, and per-cluster current directions $u_k^{(t)}$ used for the penalty.

---

## Stage-aware residuals (what is being penalized)

For each cluster $k$, we form a **partial basis up to the current stage** and measure the squared residual after projection:

* **PPCARepresentation**

  * Center: $X_c = X - \mu_k$.
  * Partial basis: $B_{k}^{(t)} = W_k[:, :t+1]$.
  * Projections: $\hat{X} = X_c B_{k}^{(t)} (B_{k}^{(t)})^\top$.
  * Residuals: $R = X_c - \hat{X}$.
  * Distance: $\mathrm{dist}_{ik} = \|R_i\|_2^2$.
  * **Current direction for penalty**: $u_k^{(t)} = W_k[:, t]$.

* **SubspaceRepresentation**

  * Identical pattern with `basis` in place of `W`.

* **Other reps**

  * Fallback to `representation.distance_to_point(points)`.

This ensures the **stage** affects distances: earlier learned columns are already “free,” and the **new column** being learned is the one under pressure from the penalty.

---

## Directional penalty (how reuse is discouraged)

`DirectionTracker.compute_penalty_batch(current_directions, penalty_type=...) → p ∈ ℝ^{n×K}`

* Input `current_directions`: tensor of shape `(n, K, d)` where

  * for each cluster $k$, the **same** current direction $u_k^{(t)}$ is broadcast across all points,
  * rows where a representation has no new column at stage $t$ can still use full-subspace distance (the fallback) but the direction is only meaningful for PPCA/Subspace.

* Output `p_{ik}`: a **scale factor** derived from the similarity between $u_k^{(t)}$ and the set of directions previously claimed by point $i$.

  * `penalty_type="product"` or `"sum"` controls **how similarities across previously claimed directions are aggregated** (examples below).
  * The strategy then mixes `p_{ik}` with 1 via $(1-\alpha)+\alpha p_{ik}$.

### Interpreting `penalty_type`

Let $S_i$ be the set of unit directions point $i$ has claimed so far; define per-direction similarity $s(u, v) = |\langle u, v\rangle|$ or $\cos\angle(u,v)$ (unit-normalized).

Two natural aggregations:

* **Product** (multiplicative inhibition):

  $$
  g_{\text{prod}}(u, S_i) \;=\; \prod_{v\in S_i} \big(1 + \lambda\,s(u,v)\big)
  $$

  Larger if $u$ aligns with **any** previously claimed directions (compounds multiplicatively).

* **Sum** (additive accumulation):

  $$
  g_{\text{sum}}(u, S_i) \;=\; 1 + \lambda\sum_{v\in S_i} s(u,v)
  $$

  Larger proportional to **total** overlap with the history.

Both produce a **scale ≥ 1** when reuse is high (bigger overlap ⇒ bigger factor). `DirectionTracker` can implement either form and return $p_{ik}=g(\,u_k^{(t)}, S_i\,)$.

> This matches the goal: **reusing** directions ⇒ $p_{ik}$ **> 1** ⇒ $\widetilde{\mathrm{dist}}_{ik}$ increases ⇒ assignment steers the point elsewhere.
> (See the “Conventions & checks” note below on ranges.)

---

## Mixing with `penalty_weight`

The code uses a **convex mixing** between “no penalty” (factor $1$) and the tracker’s scale $p$:

$$
m_{ik} \;=\; (1-\alpha) + \alpha\,p_{ik}, \qquad
\widetilde{\mathrm{dist}}_{ik} = \mathrm{dist}_{ik}\cdot m_{ik}.
$$

* $\alpha=0$ ⇒ $m_{ik}=1$ (plain distances).
* $\alpha=1$ ⇒ $m_{ik}=p_{ik}$ (full penalty).
* If $p_{ik}\ge 1$, increasing $\alpha$ **tightens** the penalty.

This is numerically stable and preserves ordering when $p\approx 1$.

---

## Returned artifacts (`aux_info`)

The method returns `(assignments, aux_info)` with:

* `distances`: raw residuals $\mathrm{dist}_{ik} \in ℝ^{n×K}$
* `penalties`: the scale $p_{ik} \in ℝ^{n×K}$
* `penalized_distances`: $\widetilde{\mathrm{dist}}_{ik} \in ℝ^{n×K}$
* `current_directions`: the broadcast $u_k^{(t)} \in ℝ^{n×K×d}$
* `current_stage`: stage index $t$
* `min_distances`: $\mathrm{dist}_{i,\widehat{k}(i)}$
* `min_penalized_distances`: $\widetilde{\mathrm{dist}}_{i,\widehat{k}(i)}$

Downstream, `KFactorsObjective` can reuse `distances` directly to avoid recomputing residuals.

---

## Algorithm sketch

```
for each cluster k:
  compute centered points Xc = X - mean_k
  choose basis B_k^(t) = columns [:t+1] (if available)
  residuals R = Xc - Xc @ B_k^(t) @ B_k^(t).T
  distances[:, k] = ||R||^2 per row
  current_directions[:, k, :] = column t of basis

if direction_tracker:
  penalties = direction_tracker.compute_penalty_batch(current_directions, penalty_type)
  scale = (1 - alpha) + alpha * penalties
  penalized = distances * scale
else:
  penalized = distances

assignments = argmin_k penalized[:, k]
return assignments, aux_info(...)
```

---

## Conventions & checks (important)

* **Direction normalization**: The design assumes columns of `W` / `basis` are **unit-norm** so cosine similarity has the intended meaning.
* **Penalty range**:

  * For the **multiplicative scaling** to **increase** cost on reuse, the tracker should return $p_{ik} \ge 1$ when overlap is high and $p_{ik}\approx 1$ when overlap is low.
  * If an implementation instead returns $p \in [0,1]$ with **smaller values meaning stronger penalty**, then multiply-by-$p$ would *reduce* the cost. In that case, redefine $p$ internally as $p' = 1 + \beta(1-p)$ (or flip the mapping) before mixing, so that higher reuse ⇒ larger $p'$.
* **Stages without a new column**: When `current_stage` exceeds available columns, the code falls back to full-subspace distance and the **direction is not meaningful**; in such cases the tracker should return $p_{ik}=1$ (neutral) for those $k$.

---

## Why this is not a heuristic

* The **residual** is exactly the squared reconstruction error after projecting onto the **partial stage basis**, aligning with subspace learning objectives.
* The **penalty** is a **deterministic functional** of angular similarity between a *candidate* direction $u_k^{(t)}$ and a *history set* $S_i$ (claimed directions).

  * `product` vs `sum` corresponds to **multiplicative** vs **additive** aggregation of similarities, both standard constructions that define a clear, differentiable (w\.r.t. $u$) scale if needed.
* The final objective used for convergence can remain the **plain residual**; the penalty only shapes the **assignment step**, steering the EM-like alternation toward **diverse directional usage** without changing the metric of record.

---

## Parameters and their effects

* `penalty_type`: choose **how** similarity accumulates across the history of a point.

  * `product`: emphasizes any **single** strong overlap (compounds quickly).
  * `sum`: emphasizes **total** overlap (more linear growth).
* `penalty_weight` $\alpha$:

  * Interpolates between **no penalty** and **full penalty**.
  * Use $\alpha\in[0.5,1]$ to make the diversity pressure appreciable; $\alpha=0$ reduces to standard K-subspaces style assignment.

---

## Practical notes

* **Complexity**: Distance evaluation is the dominant $O(nKd)$ work. Penalty evaluation is $O(nK\cdot |S_i|)$ inside the tracker if it compares the current $u_k^{(t)}$ to each stored direction $v\in S_i$; batching and caching norms keep it efficient.
* **Numerical stability**: Clamp similarities to $[0,1]$, and if using products, cap the number of factors or take logs to avoid overflow (the tracker can implement this internally).
* **Traceability**: The returned `aux_info` enables diagnostics (e.g., visualize how penalties re-rank clusters relative to raw residuals at each stage).

---

If you share `DirectionTracker.compute_penalty_batch`, I can lock down the exact formulas for `product` and `sum` to match your implementation line-by-line.
