# Weighted Subspace Credit

**Goal.** For each point $x_i$ and stage $t$, compute an **update weight** in $[0,1]$ for a candidate new direction $u$ (unit vector), reflecting how much of $u$’s “credit” has **not yet been claimed** by the point’s previously claimed directions.

* If $u$ is **orthogonal** to all previously claimed directions → weight $= 1$ (full credit).
* If $u$ lies **entirely in the span** of previously claimed directions → weight $= 0$ (no remaining credit).
* If some prior claims were weak/zero → they eat proportionally less credit, even if aligned.

This matches the “eat the credit along the feature normal” intent: claimed components remove their portion; only unclaimed components contribute to learning a new direction.

---

## Notation

For a given point $i$:

* Previously claimed (unit) directions: $d_{i,1}, \dots, d_{i,M} \in \mathbb{R}^d$.
* Their associated **assignment weights** (from earlier stages): $a_{i,1}, \dots, a_{i,M} \in [0,1]$.
  These are exactly the per-point weights you used to update those earlier directions (1 = fully claimed, 0 = ignored).
* Candidate new unit direction at the current stage: $u \in \mathbb{R}^d$, $\|u\|=1$.

---

## Definition (Weighted Subspace Credit)

Form a **weighted direction matrix**:

$$
B_i \;=\; 
\begin{bmatrix}
\sqrt{a_{i,1}}\, d_{i,1} & \sqrt{a_{i,2}}\, d_{i,2} & \cdots & \sqrt{a_{i,M}}\, d_{i,M}
\end{bmatrix}
\in \mathbb{R}^{d \times M}.
$$

Let $Q_i \in \mathbb{R}^{d \times r_i}$ be an **orthonormal basis** for the column space of $B_i$ (obtained via thin QR on $B_i$; $r_i \le M$).

Then the **remaining credit** (update weight) for direction $u$ is:

$$
w_i(u) \;=\; 1 \;-\; \|Q_i^\top u\|_2^2 \;\;\in\; [0,1].
$$

**Interpretation.** $\|Q_i^\top u\|^2$ is the fraction of $u$’s energy that lies in the **weighted span** of previously claimed directions. Subtracting from 1 returns the **unclaimed fraction**.

**Edge cases.**

* If $M=0$ (no prior claims), define $w_i(u) = 1$.
* If some $a_{i,m}=0$, its column contributes nothing to $B_i$, i.e., it **eats zero credit** even if directions are aligned.
* Clamp numerically: `w = torch.clamp(1 - (Q.T @ u).pow(2).sum(), 0.0, 1.0)`.

---

## Why this is the right signal

* **Orthogonal to all claimed:** $Q_i^\top u = 0 \Rightarrow w=1$.
* **In the claimed span:** $u \in \mathrm{span}(Q_i)\Rightarrow \|Q_i^\top u\|^2=1 \Rightarrow w=0$.
* **Partial overlap:** Credit decreases **continuously** with alignment and magnitude of prior weights.
* **Accounts for dependencies:** Unlike product/mean of $(1-|\langle u, d\rangle|)$, this method **correctly handles non-orthogonal** claimed directions via subspace geometry.

---

## Example (2D)

Let $f_0=(1,0)$, $f_1=(0,1)$, $f_2=\tfrac{1}{\sqrt2}(1,1)$, and assume $a_{i,0}=a_{i,1}=1$.

Then $B=[f_0, f_1]$, QR gives $Q=I_2$.
So $w_i(f_2) = 1 - \|Q^\top f_2\|^2 = 1 - \|f_2\|^2 = 0$.
**All credit is already claimed** by $f_0,f_1$.

If instead $a_{i,0}=1$, $a_{i,1}=0$:
$B=[f_0, \mathbf{0}]\Rightarrow \mathrm{span}(Q)=\mathrm{span}(f_0)$.
$\|Q^\top f_2\|^2 = (f_2\cdot f_0)^2 = 0.5\Rightarrow w_i(f_2)=0.5$.
**Half the credit remains** (the component orthogonal to $f_0$).

---

## Implementation Sketch

### 1) Extend `DirectionTracker` to store weights

Store `(direction, weight)` pairs per point, per stage.

```python
# Pseudocode
class DirectionTracker:
    # claimed_directions[i] : list of (Tensor[d], float)
    claimed_directions: List[List[Tuple[Tensor, float]]]

    def add_claimed_directions_batch(self, assignments, cluster_bases, claimed_weights):
        # assignments: (n,)
        # cluster_bases: (K, R, d) or (K, d) with current stage in self.current_stage
        # claimed_weights: (n,) weights in [0,1] to associate with the claimed direction
        for i in range(self.n_points):
            k = assignments[i].item()
            if cluster_bases.dim() == 3:
                direction = cluster_bases[k, self.current_stage]  # (d,)
            else:
                direction = cluster_bases[k]  # (d,)
            w = float(claimed_weights[i].item())
            self.claimed_directions[i].append((direction.detach(), w))
```

> If you don’t have `claimed_weights` yet (e.g., first stage), use `torch.ones(n)`.

### 2) Compute weights with a weighted QR

Add a new mode (e.g., `penalty_type='subspace'`) in the tracker:

```python
def compute_weighted_subspace_credit(self, test_directions: Tensor) -> Tensor:
    """
    Args:
        test_directions: (n, K, d) candidate unit directions (one per point–cluster)
    Returns:
        weights: (n, K) in [0,1]
    """
    n, K, d = test_directions.shape
    out = torch.ones(n, K, device=test_directions.device)

    for i in range(n):
        history = self.claimed_directions[i]
        if not history:
            continue  # all ones
        # Build B_i = [sqrt(w_m)*d_m] with unit d_m
        cols = []
        for (dm, wm) in history:
            if wm <= 0.0:  # ignore
                continue
            dm_u = dm / (dm.norm() + 1e-12)  # ensure unit
            cols.append((wm ** 0.5) * dm_u)
        if not cols:
            continue
        B = torch.stack(cols, dim=1)  # (d, M_eff)
        # Orthonormal basis Qi via thin QR
        Qi, _ = torch.linalg.qr(B, mode='reduced')  # (d, r)
        # For each cluster k, weight = 1 - ||Qi^T u||^2
        Ui = test_directions[i]  # (K, d)
        proj_energy = (Ui @ Qi).pow(2).sum(dim=1)  # (K,)
        wi = 1.0 - proj_energy
        out[i] = wi.clamp_(0.0, 1.0)
    return out
```

**Performance notes.**

* This loop is per point; typical $M$ (stages) is small, so QR is cheap.
* If needed, you can **cache** $Q_i$ per point and update it **incrementally** each stage by orthogonalizing the new column $\sqrt{a} d$ against existing $Q_i$ (Gram–Schmidt + re-orthonormalize).

### 3) Use the weight as the **update weight**

In the K-Factors fit loop, after assignments, gather the per-point **weight for the assigned cluster** and pass it to the updater (you already do this):

```python
# penalties = weights in [0,1], e.g., from compute_weighted_subspace_credit
stage_weights = penalties.gather(1, assignments.view(-1, 1)).squeeze(1)  # (n,)

# Then for each cluster k:
cluster_weights = stage_weights[cluster_indices]
update_strategy.update(
    representation, cluster_points,
    assignment_weights=cluster_weights,  # per-point weights
    current_stage=stage
)
```

Ensure the updater treats `assignment_weights` as **soft weights**:

* Weighted mean: $\mu = \sum w_i x_i / \sum w_i$
* Weighted residual covariance / SVD: scale centered rows by $\sqrt{w_i}$

---

## Optional: Fast Scalar Approximation

When you want to avoid QR, a cheap approximation is:

$$
\tilde{w}_i(u) \;=\; \max\!\Big(0,\; 1 - \sum_{m=1}^{M} a_{i,m}\, \langle u, d_{i,m}\rangle^2 \Big).
$$

* **Exact** if $\{d_{i,m}\}$ are orthonormal.
* **Approximate** otherwise (can over/under-count due to correlations).
* Always clamp to $[0,1]$.
* Good fallback for very large $d$ or if you’re GPU-bound.

---

## Properties

* **Range**: $w_i(u)\in[0,1]$.
* **Monotonicity**: Increasing any prior weight $a_{i,m}$ or increasing alignment $|\langle u,d_{i,m}\rangle|$ **decreases** $w_i(u)$.
* **Invariance**: Choice of basis for the claimed subspace doesn’t matter; only the **span** matters.
* **Consistency with intent**: Previously claimed components “eat” their share; only the orthogonal remainder contributes.

---

## Complexity

Per point $i$:

* Build $B_i$: $O(d\,M)$.
* Thin QR: $O(d\,M^2)$ (small $M$ makes this cheap).
* Per candidate $u$ (K clusters): projection cost $O(K\,d\,r_i)$ (often $r_i \le M \ll d$).

You can cache $Q_i$ and update incrementally each stage to reduce cost to $O(d\,r_i)$ per point per stage.

---

## Testing Ideas (later)

1. **Orthogonality test**: If history is orthonormal and full-rank in a subspace, any $u$ inside that subspace gets weight 0; orthogonal $u$ gets 1.
2. **Weight gating**: If a prior stage had $a=0$, it should not affect weights even if directions align perfectly.
3. **2D sanity**: With $f_0=(1,0), f_1=(0,1)$ (both weight 1), any $u$ has weight 0; with only $f_0$ claimed (weight 1), a 45° diagonal has weight 0.5.

---

## Summary

Weighted subspace credit uses a **weighted subspace projection** to compute how much “new” signal a candidate direction has for a point, honoring both **angles** and **prior assignment strengths**. It’s geometrically correct, numerically stable, and maps directly to the intended “credit eating” semantics:

$$
\boxed{ \; w_i(u) \;=\; 1 \;-\; \|Q_i^\top u\|_2^2 \; }
$$

where $Q_i$ spans $\mathrm{span}\{\sqrt{a_{i,m}} d_{i,m}\}$.
