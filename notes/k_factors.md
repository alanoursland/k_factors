# K-Factors algorithm

## 1. Problem Statement

Given

* a data matrix $X=\{x_1,\dots,x_n\}\subset\R^d$,
* the number of clusters $K$,
* and a target subspace dimension $R$ per cluster,

we wish to partition the points into $K$ clusters and, for each cluster $k$, extract an **orthonormal basis** $\{v_k^{(1)},\dots,v_k^{(R)}\}\subset\R^d$ (plus a mean $\mu_k$) so that each point incrementally “claims” one dimension that best explains its residual variance.

---

## 2. Algorithm Overview

At each **stage** $t=1,\dots,R$ we:

1. **Assign** each point $x_i$ to the single best cluster-direction $(k,\ell)$.
2. **Update** each cluster’s mean $\mu_k$ by averaging its assigned points.
3. **Extract** the new direction $v_k^{(t)}$ by doing a rank-1 PCA on the **residuals** after removing previously claimed directions.
4. **Record** for each point which direction it claimed, so we can penalize re-use.

---

## 3. Pseudocode

```plaintext
Input: X ∈ ℝ^{n×d},  K,  R
Output:  μ₁…μ_K ∈ ℝ^d;  for each k,  {v_k^(1)…v_k^(R)} orthonormal in ℝ^d;
        and per-point claimed directions D_i = [d_i¹…d_i^R]

Initialize all μ_k (e.g. by K-means centroids)
For all i,  set D_i←empty list

for t = 1 to R:
  // ——————————————————————————————
  // 1) (Hard) Assignment Step with Penalty
  for i in 1…n:
    for k in 1…K:
      // penalty from previously claimed directions
      p_{ik} = ∏_{d∈D_i} (1 − |v_k·d|)
      // residual projector after removing old directions in cluster k
      P_k^{(<t)} = ∑_{u=1}^{t-1} v_k^{(u)} v_k^{(u)ᵀ}
      r_{ik} = (I − P_k^{(<t)}) (x_i − μ_k)
      cost_{ik} = p_{ik} · ‖r_{ik}‖²
    end
    (k_i,ℓ_i) = argmin_{k} cost_{ik}          // ℓ_i ≡ t implicitly
    assign x_i → cluster k_i
  end

  // 2) Update Means
  for k in 1…K:
    let 𝒳_k = { x_i : i assigned to k }
    μ_k ← (1/|𝒳_k|) ∑_{x∈𝒳_k} x
  end

  // 3) Sequential PCA on Residuals → new v_k^(t)
  for k in 1…K:
    Form residuals for assigned points:
      y_i = x_i − μ_k
      r_i = (I − ∑_{u=1}^{t-1} v_k^(u) v_k^(u)ᵀ ) y_i
    Compute scatter S_k = ∑_{i:assigned} r_i r_iᵀ
    v_k^(t) = top eigenvector of S_k   // unit‐norm
  end

  // 4) Record claimed directions
  for each point x_i:
    append d_i^t ← v_{k_i}^{(t)}  to D_i
  end
end
```

---

## 4. Time Complexity

Let $n$ = #points, $d$ = ambient dimension, $K$ = #clusters, $R$ = subspace dimension per cluster, and assume we run **one assignment + mean-update** cycle per stage (for simplicity).

1. **Assignment step** per stage $t$:

   * For each point–cluster pair $(i,k)$:

     * Compute penalty $p_{ik}$: $O(t)$ inner products.
     * Compute residual $r_{ik}$: projecting out $t-1$ directions costs $O(t\,d)$.
     * Squared norm: $O(d)$.
   * Total per stage:

     $$
       O\bigl(n\;K\;(t\,d)\bigr)
       \;\le\;
       O(n\,K\,d\,R).
     $$

2. **Mean update** per stage:

   $$
     O(n\,d).
   $$

3. **Sequential PCA** per cluster per stage:

   * Residual scatter: $O(|\mathcal X_k|\,d\,t)$, summing over $k$ gives $O(n\,d\,t)$.
   * Top‐eigenvector of a $d\times d$ matrix: $O(d^2)$ per power-iteration; constant number ⇒ $O(d^2)$.
   * Over $K$ clusters:

     $$
       O(n\,d\,t \;+\; K\,d^2)
       \;\le\;
       O(n\,d\,R + K\,d^2).
     $$

Summing across $t=1\ldots R$, the **total** time is on the order of

$$
\sum_{t=1}^R \Bigl[O(n\,K\,d\,t) + O(n\,d) + O(n\,d\,t + K\,d^2)\Bigr]
\;=\;
O\bigl(n\,K\,d\,R^2 + K\,d^2\,R + n\,d\,R\bigl).
$$

In practice $R\ll d$ or $R\ll K$, so the dominant costs are
$\;O(n\,K\,d\,R^2)$ for assignment and
$\;O(K\,d^2\,R)$ for eigen‐computations.

---

## 5. Space Complexity

* **Data matrix** $X$: $O(n\,d)$.
* **Cluster means** $\{\mu_k\}$: $O(K\,d)$.
* **Directions** $\{v_k^{(u)}\}$: $O(K\,d\,R)$.
* **Point‐wise claimed lists** $D_i$: storing $t$ unit-vectors per point ⇒ $O(n\,d\,R)$.
* **Temporary residuals/scatters**: $O(d^2)$ per cluster.

Overall:

$$
O\bigl(n\,d \;+\; K\,d\,R \;+\; n\,d\,R\bigr).
$$

---

### Remarks

* This algorithm is **block-coordinate-descent** on a well-defined objective (hard Mixture of PPCA), so it **monotonically decreases** that objective.
* In the special case $R=1$ it reduces to the standard **K-Lines / K-Subspaces** algorithm.
* When soft assignments replace the hard arg-mins, you recover **Mixtures of Probabilistic PCA** (an EM algorithm).

