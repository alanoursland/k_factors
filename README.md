# K‑Factors & C‑Factors — Exploratory Notes

> **Status: Early & Exploratory**
> These documents and the accompanying code capture ongoing research ideas. APIs, algorithms, and names may change.

-----

## What is this project?

We are experimenting with **K‑Factors**, a hard‑assignment, sequential subspace–clustering algorithm that extends K‑Means and K‑Subspaces by letting each cluster grow an *orthogonal* low‑rank basis one dimension at a time. A soft‑assignment counterpart — **C‑Factors** — is simply the classic *Mixture of Probabilistic PCA* (a.k.a. Mixture of Factor Analyzers), renamed here for pedagogical symmetry with Fuzzy C‑Means.

At a glance:

| Method        | Assignment | Cluster Representation   | Goal                            |
|---------------|------------|--------------------------|---------------------------------|
| **K‑Factors** | Hard       | Sequential $R$-dim basis | Fast, deterministic subspace clustering |
| **C‑Factors** | Soft (EM)  | Mixture of PPCA          | Probabilistic latent‑factor model     |

Our long‑term motivation is to use the learned local bases for tasks such as neural‑network weight initialization, data compression, and anomaly detection.

## Getting Started

As this is an exploratory project, the code is not yet packaged for distribution. To experiment with the algorithms, clone the repository and install the necessary dependencies.

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies (assuming a requirements.txt file will be added)
pip install -r requirements.txt
```

## Code Architecture & Status

The `src/` directory contains the Python implementation of the algorithms. The code is designed to be modular, allowing for different components of the clustering process to be combined and extended.

The core logic is found within the `kfactors/` package, which is organized as follows:

  * **`kfactors/algorithms/`**: Contains the main clustering algorithms.

      * **Implemented**: `kmeans.py`, `kfactors.py`
      * **Planned**: `cfactors.py`, `gmm.py`, `ksubspaces.py`

  * **`kfactors/base/`**: Provides the foundational building blocks for all algorithms. This includes a `ClusteringBase` class, common `DataStructures`, and abstract `Interfaces` that ensure a consistent API.

  * **`kfactors/representations/`**: Defines how each cluster is mathematically represented.

      * **Implemented**: `centroid.py` (for K-Means), `subspace.py` (for K-Factors), and `ppca.py` (the core model for C-Factors).
      * **Planned**: `gaussian.py` (for GMMs).

  * **`kfactors/assignments/`**: Handles the logic for assigning data points to clusters.

      * **Implemented**: `hard.py` (for K-Means/K-Factors) and `penalized.py` (for custom assignment costs).
      * **Planned**: `soft.py` and `fuzzy.py` for probabilistic models.

  * **`kfactors/updates/`**: Contains the rules for updating a cluster's representation.

      * **Implemented**: `mean.py` (for centroids) and `sequential_pca.py` (the core "growth" logic for K-Factors).

  * **`kfactors/initialization/`**: Provides methods for selecting the initial cluster seeds.

      * **Implemented**: `kmeans_plusplus.py`

  * **`kfactors/utils/`**: A collection of helper modules for linear algebra (`linalg.py`), convergence checking (`convergence.py`), and more.

### Current Implementation Status

As of now, the framework supports a fully functional **K-Means** algorithm and the core logic for the **K-Factors** algorithm. While the representational model for C-Factors (`ppca.py`) is implemented, the full end-to-end C-Factors algorithm is still in development. Demos, visualization tools, and other algorithms like GMM are planned for the future.

## Research Notes

The `notes/` directory contains the theoretical background, performance comparisons, and historical context for this work.

```
notes/
  k_factors.md              # Algorithm, pseudocode, complexity
  c_factors.md              # Soft EM variant (Mixture of PPCA)
  comparative_performance.md  # Runtime & memory comparison tables
  prior_art.md              # Historical context & citations
  applications.md           # Practical use‑cases
```

### How to Read These Notes

1.  **Start with `notes/k_factors.md`** for the core hard‑assignment algorithm.
2.  **Skim `notes/c_factors.md`** to see how it reduces to the well‑known Mixture of PPCA.
3.  **Check `notes/comparative_performance.md`** to understand where K‑Factors sits cost‑wise between K‑Means and full GMMs.
4.  **Browse `notes/prior_art.md`** if you need citations or historical perspective.
5.  **See `notes/applications.md`** for why we think this matters.

## Contributing

This is an active research project, and we welcome contributions. If you are interested in implementing one of the planned features, fixing a bug, or suggesting an improvement, please feel free to open an issue or submit a pull request.

## Acknowledgements

Conceptual roots include MacQueen’s K‑Means, Bradley & Mangasarian’s K‑Planes, Tipping & Bishop’s Probabilistic PCA, and Ghahramani & Hinton’s Factor Analyzers. This repo re‑frames and combines these ideas in search of fast, practical latent‑factor learning.

