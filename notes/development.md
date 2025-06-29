# Theoretical Foundations and Development of K-Factors

This document traces the evolution from basic clustering and latent‐factor models to our **K‑Factors** algorithm, grounding each step in a clear block‑coordinate‐descent or EM framework.

---

## 1. Block‑Coordinate Descent: A Unifying Perspective

Many clustering and subspace‐learning algorithms can be viewed as **alternating minimization** (hard assignments) or **EM** (soft assignments) on a well‑defined objective.  At each iteration, you:

1. **Assignment (E‑step or hard assign):** Fix model parameters, update cluster memberships or responsibilities.
2. **Update (M‑step or centroid/subspace fit):** Fix assignments, optimize parameters to minimize expected loss or maximize expected log‑likelihood.

This guarantees monotonic improvement of the objective and finite convergence (to a local optimum in the hard case).

---

## 2. K‑Means: Hard Assignments on SSE

**Objective:**

$$
J(\{C_k\},\{\mu_k\}) = \sum_{k=1}^K \sum_{x_i\in C_k} \|x_i - \mu_k\|^2.
$$

**Block steps:**

* **Assignment:**  $c_i = \arg\min_k\|x_i-\mu_k\|^2$.
* **Update:**  $\mu_k = \frac1{|C_k|}\sum_{i:c_i=k}x_i$.

Monotonic decrease follows because each step optimally solves its subproblem.  K‑Means is the **hard‑EM** limit of an isotropic GMM as variance →0.

---

## 3. Fuzzy C‑Means: Soft Assignments on SSE

**Objective:**

$$
J = \sum_{i=1}^n\sum_{k=1}^K u_{ik}^m\|x_i-\mu_k\|^2,\quad u_{ik}\in[0,1],\;\sum_k u_{ik}=1.
$$

**Updates:**

* **Membership:**
  $u_{ik} = \frac{\|x_i-\mu_k\|^{-2/(m-1)}}{\sum_j\|x_i-\mu_j\|^{-2/(m-1)}}$.
* **Centroids:**
  $\mu_k = \frac{\sum_i u_{ik}^m x_i}{\sum_i u_{ik}^m}$.

As $m\to1$, memberships become hard (0/1), recovering K‑Means.

---

## 4. Isotropic GMM: Soft EM with Shared Spherical Covariance

**Model:** $x_i\sim\sum_k\pi_k\mathcal N(\mu_k,\sigma^2I)$.
**E‑step:** responsibilities $r_{ik}\propto \pi_k\exp(-\|x_i - \mu_k\|^2/(2\sigma^2))$.
**M‑step:** centroid updates as in fuzzy C‑means, plus $\sigma^2 = \frac1{nd}\sum_{i,k}r_{ik}\|x_i-\mu_k\|^2$.

As $\sigma^2\to0$, assignments harden and recovery of K‑Means occurs.

---

## 5. Full‑Covariance GMM: Soft EM with $\Sigma_k$

**Model:** $x_i\sim\sum_k\pi_k\mathcal N(\mu_k,\Sigma_k)$.
**E‑step:** $r_{ik}\propto\pi_k|\Sigma_k|^{-1/2}\exp(-\tfrac12(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k))$.
**M‑step:** update $\mu_k,\Sigma_k,\pi_k$ via weighted ML.

**Hard‑EM** variant uses $r_{ik}\in\{0,1\}$.  With $\Sigma_k$ unconstrained, this generalizes K‑Means to ellipsoidal clusters.

---

## 6. K‑Subspaces / K‑Lines: Hard‑Assign 1D Subspaces

**Objective:** minimize sum of squared orthogonal distances to cluster subspaces.  For $r=1$:

$$
J = \sum_{k=1}^K \sum_{x_i\in C_k} \|(I - v_kv_k^T)(x_i - \mu_k)\|^2.
$$

**Steps:**

* **Assignment:** nearest line via orthogonal distance.
* **Update:** $\mu_k = \text{mean}(C_k)$, and $v_k$=top eigenvector of scatter of $C_k$.

This is hard‑EM on a rank‑1 Gaussian mixture (zero noise on subspace).

---

## 7. Mixture of PPCA (C‑Factors): Soft EM with Low‑Rank Covariances

**Model:** $x_i|z=k\sim \mathcal N(\mu_k, W_kW_k^T + \sigma_k^2I)$, soft assignments.
**E‑step:** compute responsibilities.
**M‑step:** update $\mu_k$, factor loadings $W_k$ (via eigen‑analysis of soft scatter), and $\sigma_k^2$ (averaged residual variance).

Renamed here as **C‑Factors** for pedagogical symmetry with Fuzzy C‑Means.

---

## 8. Hard Mixtures of PPCA: K‑Factors

**Hard assignments** + **sequential dimension claiming** yields:

1. **Assignment:** each point picks one new direction $v_k^{(t)}$ with penalty on re‑use.
2. **Update:** sequential PCA on residuals per cluster to extract $v_k^{(t)}$.
3. Repeat for $t=1\dots R$.

This performs block‑coordinate‐descent on a hard PPCA objective, extracting an orthonormal basis of size $R$ per cluster.

---

## 9. Summary of Development Path

$$
\text{K‑Means}\;(\mu) \;\to\; \text{Fuzzy C‑Means}\;(u) \;\to\; \text{Isotropic GMM}\;(\sigma^2) \;\to\; \text{Full GMM}\;(\Sigma) \\
\to\; \text{K‑Lines (r=1)} \;(v) \;\to\; \text{C‑Factors (PPCA mixture)} \;(W,\sigma^2) \;\to\; \text{K‑Factors} \;(\text{hard }W)
$$

Each arrow corresponds to either **soft→hard** in assignments or **increasing subspace complexity** in representation.  K‑Factors arise naturally as the **hard** counterpart of C‑Factors, combining the efficiency of block‑coordinate‑descent with the expressiveness of local latent bases.

---

*End of derivation.*
