# C-Factors (Soft Mixtures of Probabilistic PCA)

> **Note:** “C-Factors” is a pedagogical **renaming** of the classic Mixture of Probabilistic PCA (also known as Mixture of Factor Analyzers) to parallel the **C-Means** naming convention. It **does not** introduce new methodology—the generative model, inference, and EM updates remain identical.

---

## 1. Problem Statement

Given:

* A data matrix $X = \{x_1, \dots, x_n\}\subset \mathbb R^d$
* Number of clusters $C$ (analogous to $K$ in hard-assignment)
* Target latent dimension $R$ per cluster

We seek a soft partitioning of the data into $C$ Gaussian components, each with its own low-dimensional latent subspace of dimension $R$.

---

## 2. Generative Model

For each cluster $k=1,\dots,C$:

1. **Mixing proportion:** $\pi_k$, with $\sum_k \pi_k = 1$.
2. **Mean:** $\mu_k\in\mathbb R^d$.
3. **Factor loading matrix:** $W_k\in\mathbb R^{d\times R}$.
4. **Isotropic noise variance:** $\sigma_k^2>0$.

Latent factors $z_i\in\mathbb R^R$ are drawn from $\mathcal N(0, I_R)$, and noise $\epsilon_i\sim\mathcal N(0,\sigma_k^2 I_d)$. The observation model:

$$
\begin{aligned}
x_i &= \mu_k + W_k z_i + \epsilon_i,\\
x_i &\sim \mathcal N\bigl(\mu_k, \;C_k\bigr),
\quad C_k = W_k W_k^T + \sigma_k^2 I_d.
\end{aligned}
$$

---

## 3. EM Algorithm (Soft Assignments)

We maximize the data log-likelihood by iterating between:

### E-Step

Compute **responsibilities** $r_{ik}$ for each point $x_i$ and component $k$:

$$
r_{ik} = P(z_i=k \mid x_i)
= \frac{\pi_k \;\mathcal N(x_i\mid\mu_k, C_k)}{\sum_{j=1}^C \pi_j\;\mathcal N(x_i\mid\mu_j, C_j)}.
$$

Compute effective counts: $N_k = \sum_{i=1}^n r_{ik}$.

### M-Step

Update parameters using the soft counts and expected sufficient statistics:

1. **Mixing proportions**

   $$
   \pi_k \leftarrow \frac{N_k}{n}.
   $$

2. **Means**

   $$
   \mu_k \leftarrow \frac{1}{N_k}\sum_{i=1}^n r_{ik}\,x_i.
   $$

3. **Factor loadings** $W_k$ and **noise** $\sigma_k^2$:

   * Compute the **soft scatter**
     $S_k = \frac{1}{N_k} \sum_{i=1}^n r_{ik} (x_i-\mu_k)(x_i-\mu_k)^T$.
   * Perform eigen-decomposition of $S_k = U_k \Lambda_k U_k^T$ (keep top $R$ eigenvalues/lambda).
   * Closed-form updates (Tipping & Bishop):

     $$
     W_k \leftarrow U_k\bigl(\Lambda_k - \sigma_k^2 I_R\bigr)^{1/2} R,\quad
     \sigma_k^2 \leftarrow \frac{1}{d-R}\sum_{j=R+1}^d \lambda_{k,j},
     $$

     where $\{\lambda_{k,j}\}$ are the eigenvalues of $S_k$ and $R$ is an arbitrary rotation (often set to identity).

Repeat E- and M-steps until convergence of the log-likelihood.

---

## 4. Relationship to **K-Factors**

* **C-Factors** = Mixture of Probabilistic PCA with **soft** cluster memberships (EM).
* **K-Factors** is the **hard-assignment** limit (each $r_{ik}\in\{0,1\}$) with a sequential dimension-claiming penalty, as documented in `k_factors.md`.

This parallel mirrors **K-Means** ↔ **C-Means**, showing how hard vs. soft assignments yield related algorithms.

---

## 5. Computational Complexity

For each EM iteration (assume $n$ data points, $d$ dims, $C$ clusters, latent dim $R$):

* **E-Step:** $O(n\,C\,(d^3 + d))$ if directly computing multivariate normals (with Cholesky), or optimized to $O(n\,C\,d\,R)$ using the Woodbury identity for low-rank covariances.
* **M-Step:** $O(n\,d\,C + C\,d^2\,R + C\,R^3)$ for scatter computation and eigen-decompositions.

This sits above **K-Factors** in time but below full GMMs with unconstrained covariances.

---

## 6. References

* Michael E. Tipping & Christopher M. Bishop. *Mixtures of Probabilistic Principal Component Analysers*. Neural Computation, 1999.
* Zoubin Ghahramani & Geoffrey E. Hinton. *The EM Algorithm for Factor Analyser Models*. Technical Report, University of Toronto, 1996.
* J. MacQueen. *Some methods for classification and analysis of multivariate observations*. 1967.
* J. C. Bezdek. *Pattern Recognition with Fuzzy Objective Function Algorithms*. 1981.
