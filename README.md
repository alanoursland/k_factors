# K‑Factors & C‑Factors — Exploratory Notes

> **Status: early & exploratory**
> These documents capture ongoing research ideas.  APIs, algorithms, and names may change.

---

## What is this project?

We are experimenting with **K‑Factors**, a hard‑assignment, sequential subspace–clustering algorithm that extends K‑Means and K‑Subspaces by letting each cluster grow an *orthogonal* low‑rank basis one dimension at a time.  A soft‑assignment counterpart — **C‑Factors** — is simply the classic *Mixture of Probabilistic PCA* (a.k.a. Mixture of Factor Analyzers), renamed here for pedagogical symmetry with Fuzzy C‑Means.

At a glance:

| Method        | Assignment | Cluster representation   | Goal                                    |
| ------------- | ---------- | ------------------------ | --------------------------------------- |
| **K‑Factors** | Hard       | Sequential $R$-dim basis | Fast, deterministic subspace clustering |
| **C‑Factors** | Soft (EM)  | Mixture of PPCA          | Probabilistic latent‑factor model       |

Our long‑term motivation is to use the learned local bases for tasks such as neural‑network weight initialization, data compression, and anomaly detection.

---

## Directory layout

```
notes/
  k_factors.md               # Algorithm, pseudocode, complexity
  c_factors.md               # Soft EM variant (Mixture of PPCA)
  comparative_performance.md # Runtime & memory comparison tables
  prior_art.md               # Historical context & citations
  applications.md            # Practical use‑cases
```

---

## How to read these notes

1. **Start with `notes/k_factors.md`** for the core hard‑assignment algorithm.
2. **Skim `notes/c_factors.md`** to see how it reduces to the well‑known Mixture of PPCA.
3. **Check `notes/comparative_performance.md`** to understand where K‑Factors sits cost‑wise between K‑Means and full GMMs.
4. **Browse `notes/prior_art.md`** if you need citations or historical perspective.
5. **See `notes/applications.md`** for why we think this matters.

---

## Acknowledgements

Conceptual roots include MacQueen’s K‑Means, Bradley & Mangasarian’s K‑Planes, Tipping & Bishop’s Probabilistic PCA, and Ghahramani & Hinton’s Factor Analyzers.  This repo re‑frames and combines these ideas in search of fast, practical latent‑factor learning.
