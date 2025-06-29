# Prior Art and Context for K-Factors

This document reviews the key algorithms and models that underpin **K-Factors**, situating it within the historical lineage of clustering and latent-space methods.

---

## 1. K-Means (MacQueen, 1967)

* **Input:** Points in $\mathbb R^d$
* **Represents:** Each cluster by its **centroid** (mean)
* **Assignment:** Hard nearest-centroid
* **Update:** Centroid = cluster mean
* **Objective:** Minimize within-cluster sum of squared errors (SSE)

Classic reference:

> J. MacQueen. *Some Methods for Classification and Analysis of Multivariate Observations*, 1967.

---

## 2. Fuzzy C-Means (Bezdek, 1981)

* **Input:** Points in $\mathbb R^d$
* **Represents:** Centroids, with **soft memberships** $u_{ik}\in[0,1]$
* **Assignment:** Soft via membership weights
* **Update:** Centroids weighted by memberships
* **Objective:** Fuzzified SSE with membership exponent

Classic reference:

> J. C. Bezdek. *Pattern Recognition with Fuzzy Objective Function Algorithms*, 1981.

---

## 3. Soft K-Means / EM for Isotropic GMM

* **Input:** Points
* **Represents:** Isotropic Gaussians ($\sigma^2 I$) at each centroid
* **Assignment:** Soft responsibilities via Gaussian likelihood
* **Update:** EM updates for means and shared variance
* **Objective:** Maximize data log-likelihood under isotropic GMM

Often called **Soft K-Means** or **EM++** in clustering literature.

---

## 4. Gaussian Mixture Models (Full-Covariance GMM)

* **Input:** Points
* **Represents:** Full-covariance Gaussians
* **Assignment:** Soft via posterior probabilities
* **Update:** EM for means, covariances, and weights
* **Objective:** Maximize full GMM log-likelihood

Widely used; can model arbitrary ellipsoidal clusters.

---

## 5. Probabilistic PCA (Tipping & Bishop, 1999)

* **Input:** Points
* **Model:** $x=\mu + Wz + \epsilon$, $z\sim\mathcal N(0,I)$, $\epsilon\sim\mathcal N(0,\sigma^2 I)$
* **Inference:** Closed-form ML for $W,\mu,\sigma^2$ via SVD
* **Output:** Global low-dimensional subspace (principal components)

Reference:

> M. E. Tipping & C. M. Bishop. *Probabilistic Principal Component Analysis*, 1999.

---

## 6. Mixture of Probabilistic PCA / Factor Analyzers (Ghahramani & Hinton, 1996; Tipping & Bishop, 1999)

* **Input:** Points
* **Represents:** Each cluster as a low-rank Gaussian (factor analyzer)
* **Assignment:** Soft via responsibilities
* **Update:** EM for $\{\pi_k,\mu_k,W_k,\sigma_k^2\}$
* **Objective:** Maximize mixture PPCA log-likelihood

References:

> Z. Ghahramani & G. Hinton. *The EM Algorithm for Factor Analyser Models*, 1996.
> M. E. Tipping & C. M. Bishop. *Mixtures of Probabilistic Principal Component Analysers*, 1999.

---

## 7. K-Subspaces / K-Planes Clustering (Bradley & Mangasarian, 2000)

* **Input:** Points in $\mathbb R^d$
* **Represents:** Each cluster by an $r$-dimensional affine subspace
* **Assignment:** Hard nearest-subspace via orthogonal distance
* **Update:** SVD/PCA to fit subspace to assigned points
* **Objective:** Minimize sum of squared orthogonal distances

Key reference:

> P. S. Bradley & O. L. Mangasarian. *k-Plane Clustering*, Journal of Global Optimization, 2000.

---

## 8. K-Lines (Special case of K-Subspaces, $r=1$)

* **Input:** Points
* **Represents:** 1-D lines per cluster
* **Assignment:** Hard via orthogonal distances to lines
* **Update:** PCA to extract first principal component
* **Objective:** Minimize orthogonal SSE to lines

Also referred to as **1-D subspace clustering**.

---

## 9. K-Factors (This Work)

* **Input:** Points
* **Represents:** Sequentially-built orthonormal basis of dimension $R$ per cluster
* **Assignment:** Hard with claimed-direction penalty
* **Update:** Sequential PCA on residuals
* **Objective:** Block-coordinate descent on a hard MoPPCA objective

K-Factors extends K-Subspaces by extracting multiple dimensions per cluster with a novel penalty to avoid reusing directions.

---

### How K-Factors Fits In

| Algorithm     | Assignment | Subspace Dim | Hard/Soft | Core Update                |
| ------------- | ---------- | ------------ | --------- | -------------------------- |
| K-Means       | Centroid   | 0            | Hard      | Mean                       |
| K-Subspaces   | Subspace   | $r$          | Hard      | PCA/SVD                    |
| Mixture PPCA  | Covariance | $R$          | Soft      | EM for $W,\sigma^2$        |
| **K-Factors** | Subspace   | $R$          | Hard      | Sequential PCA (residuals) |

K-Factors is effectively the **hard-assignment limit** of Mixture of PPCA, paralleling how K-Means relates to isotropic GMMs and Fuzzy C-Means.
