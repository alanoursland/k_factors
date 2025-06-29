# Comparative Performance of K-Factors and Related Algorithms

This document summarizes the runtime and memory costs of **K-Factors** (hard-assignment sequential PPCA) alongside several related clustering and latent-factor methods. It highlights how **K-Factors** trades off efficiency and modeling power compared to baselines and full probabilistic approaches.

---

## 1. Time Complexity Comparison

| Algorithm                           | Time per Iteration             | Notes                                                                      |
| ----------------------------------- | ------------------------------ | -------------------------------------------------------------------------- |
| **K-Means**                         | $O(n K d)$                     | Pure point-to-centroid squared distances                                   |
| **K-Lines** (r=1 K-Subspaces)       | $O(n K d + K d^2)$             | Adds 1-D PCA per cluster ($d^2$ eigenvector cost)                          |
| **K-Factors** (hard MoPPCA, R dims) | $O(n K d R^2 + K d^2 R)$       | Sequential PCA & penalty in assignment; $R\ll d$ typical                   |
| **C-Factors** (Mixture of PPCA, EM) | $O(n C d R + C d^2 R + C R^3)$ | Soft responsibilities + low-rank covariance updates; Woodbury can optimize |
| **GMM (full cov.)**                 | $O(n K d^2 + K d^3)$           | Inverting/decomposing full $d\times d$ covariances per component           |

**Legend:**

* $n$: number of data points
* $d$: ambient dimension
* $K,C$: number of clusters (hard vs. soft)
* $R$: target subspace dimension per cluster ($R=1$ for K-Lines)

---

## 2. Space Complexity Comparison

| Algorithm          | Memory Footprint         | Notes                                            |
| ------------------ | ------------------------ | ------------------------------------------------ |
| **K-Means**        | $O(n d + K d)$           | Data + centroids                                 |
| **K-Lines**        | $O(n d + K d + K d)$     | Data + centroids + single direction per cluster  |
| **K-Factors**      | $O(n d + n d R + K d R)$ | Data + per-point claimed vectors + cluster bases |
| **C-Factors** (EM) | $O(n d + C d R)$         | Data + loading matrices                          |
| **GMM**            | $O(n d + K d^2)$         | Data + full covariances                          |

---

## 3. Practical Trade-offs

* **K-Means** is the fastest and simplest but only models spherical clusters.
* **K-Lines** (single dimension) adds minimal cost for one direction per cluster.
* **K-Factors** (sequential PCA) costs grow **quadratically** in $R$ during assignment, but remain **faster** than full EM for moderate $R\ll d$ or small $K$.
* **C-Factors** (EM Mixture of PPCA) offers soft membership and probabilistic interpretation; more expensive in the E-step but benefits from low-rank optimizations (Woodbury) in M-step.
* **Full GMMs** are most expressive but become prohibitive when $d$ is large due to $d^3$ covariance operations.

---

## 4. When to Choose Which

* **Use K-Means** for very large $n,d$ when clusters are roughly spherical.
* **Use K-Lines** if you need one principal direction per cluster cheaply (e.g. PCA within clusters).
* **Use K-Factors** when you want **multiple sequential dimensions** per cluster with **hard assignments** and faster than EM.
* **Use C-Factors** to get a **probabilistic** soft-clustering with low-rank subspace structure.
* **Use full GMMs** only when you need **arbitrary** covariance shapes and $d$ is moderate.

---

*All complexities assume a constant number of power-iterations for eigenvector extraction and no specialized hardware.*
