# K‑Factors: A Unified Framework for Clustering with Local Subspaces

> **Status: Research Implementation**  
> This repository contains a comprehensive PyTorch implementation of K-Factors and related clustering algorithms. While the code is feature-complete and tested, APIs may evolve based on research needs and community feedback.

## Overview

K-Factors is a novel hard-assignment clustering algorithm that extends K-Means and K-Subspaces by sequentially extracting orthogonal basis vectors for each cluster. The soft-assignment counterpart, C-Factors, is the classic Mixture of Probabilistic PCA (MoPPCA), renamed for pedagogical symmetry with the K-Means/C-Means naming convention.

### Algorithm Family at a Glance

| Method        | Assignment | Cluster Representation   | Key Innovation                           |
|---------------|------------|--------------------------|------------------------------------------|
| **K-Means**   | Hard       | Centroids               | Simple, fast, well-understood            |
| **K-Factors** | Hard       | Sequential R-dim basis  | Penalty mechanism prevents direction reuse |
| **C-Factors** | Soft (EM)  | Mixture of PPCA        | Classic MoPPCA with pedagogical renaming |
| **GMM**       | Soft (EM)  | Full Gaussians         | Flexible covariance structures           |

### Theoretical Foundation

The library implements clustering algorithms across two main dimensions:

```
                    Hard Assignment              Soft Assignment
                         |                            |
Centroids:          K-Means                    Fuzzy C-Means
                         |                            |
Subspaces:          K-Lines/K-Subspaces       (via soft assignment)
                         |                            |
Sequential:         K-Factors                  C-Factors (Mixture of PPCA)
                         |                            |
Full Gaussian:      (via hard assignment)      Gaussian Mixture Model
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kfactors.git
cd kfactors

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install torch numpy matplotlib scikit-learn
```

## Quick Start

### Basic Usage Pattern

All algorithms follow the sklearn pattern:

```python
import torch
from kfactors import KMeans, KFactors, GaussianMixture

# Generate sample data
X = torch.randn(1000, 20)

# Fit any algorithm
model = KMeans(n_clusters=5)  # or any other algorithm
model.fit(X)

# Get cluster assignments
labels = model.predict(X)

# Get cluster centers (available for all algorithms)
centers = model.cluster_centers_
```

### GPU Acceleration

```python
# Automatic GPU usage if available
model = KMeans(n_clusters=5, device='auto')

# Explicit device selection
model = KMeans(n_clusters=5, device='cuda:0')

# CPU only
model = KMeans(n_clusters=5, device='cpu')
```

## Core Algorithms

### 1. K-Means

Standard centroid-based clustering:

```python
from kfactors import KMeans

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',  # or 'random' or custom centers
    max_iter=100,
    tol=1e-4,
    verbose=1
)
kmeans.fit(X)

# Results
labels = kmeans.labels_
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_
```

### 2. K-Factors (Novel Algorithm)

Sequential subspace extraction with penalty mechanism:

```python
from kfactors import KFactors

# Key innovation: sequential extraction prevents direction collapse
kfactors = KFactors(
    n_clusters=4,
    n_components=3,  # dimensions extracted sequentially
    penalty_weight=0.9,  # controls direction reuse penalty
    max_iter=30
)
kfactors.fit(X)

# Inspect stage-wise progress
for stage in kfactors.stage_history_:
    print(f"Stage {stage['stage']}: {stage['iterations']} iterations")

# Access the orthonormal bases
bases = kfactors.cluster_bases_  # (n_clusters, n_features, n_components)
```

### 3. C-Factors (Mixture of PPCA)

Soft clustering with low-rank Gaussian models:

```python
from kfactors import CFactors

# This is classical Mixture of PPCA, renamed for consistency
cfactors = CFactors(
    n_clusters=4,
    n_components=3,  # latent dimensions
    max_iter=100
)
cfactors.fit(X)

# Transform to latent space
Z = cfactors.transform(X)

# Get soft assignments
responsibilities = cfactors.predict_proba(X)
```

### 4. K-Subspaces

Clusters data along r-dimensional affine subspaces:

```python
from kfactors import KSubspaces

# Fit 2D subspaces in 10D data
ksub = KSubspaces(
    n_clusters=4,
    n_components=2,  # subspace dimension
    max_iter=50
)
ksub.fit(X)

# Transform to local coordinates
X_local = ksub.transform(X)  # (n_samples, n_components)

# Reconstruct from local coordinates
X_reconstructed = ksub.inverse_transform(X_local)
```

### 5. Gaussian Mixture Model

Full probabilistic clustering with various covariance types:

```python
from kfactors import GaussianMixture

gmm = GaussianMixture(
    n_clusters=3,
    covariance_type='full',  # 'diagonal', 'spherical', 'tied'
    n_init=5,  # number of initializations
    max_iter=100
)
gmm.fit(X)

# Get soft assignments (probabilities)
probs = gmm.predict_proba(X)

# Generate new samples
X_new, y_new = gmm.sample(100)

# Model selection
bic = gmm.bic(X)
aic = gmm.aic(X)
```

### 6. Fuzzy C-Means

Soft clustering with fuzzy memberships:

```python
from kfactors import FuzzyCMeans

fcm = FuzzyCMeans(
    n_clusters=4,
    m=2.0,  # fuzziness parameter (m>1)
    max_iter=100
)
fcm.fit(X)

# Get fuzzy memberships
memberships = fcm.u_  # (n_samples, n_clusters)

# Evaluate fuzziness
fpc = fcm.partition_coefficient()  # closer to 1 = less fuzzy
fpe = fcm.partition_entropy()  # lower = less fuzzy
```

## Research Applications

The learned local bases from K-Factors and C-Factors are particularly useful for:

1. **Neural Network Weight Initialization**: Use cluster bases as informed priors
2. **Data Compression**: Piecewise linear encoding with local coordinates
3. **Anomaly Detection**: Points with large residuals indicate anomalies
4. **Transfer Learning**: Learn bases on source domain, adapt on target
5. **Interpretable Dimensionality Reduction**: Each cluster has its own interpretable subspace

## Evaluation and Metrics

### Internal Metrics (no ground truth)

```python
from kfactors.utils.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Fit model
model = KMeans(n_clusters=5)
model.fit(X)
labels = model.labels_

# Evaluate
sil = silhouette_score(X, labels)  # higher is better [-1, 1]
db = davies_bouldin_score(X, labels)  # lower is better
ch = calinski_harabasz_score(X, labels)  # higher is better
```

### External Metrics (with ground truth)

```python
from kfactors.utils.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)

# Compare with true labels
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
homo, comp, v_score = v_measure_score(true_labels, pred_labels)
```

## Visualization

### Basic 2D/3D Visualization

```python
from kfactors import plot_clusters_2d, plot_clusters_3d
import matplotlib.pyplot as plt

# For 2D data
plot_clusters_2d(X_2d, labels, centers=centers)
plt.show()

# For 3D data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_clusters_3d(X_3d, labels, centers=centers, ax=ax)
plt.show()
```

### Fuzzy Clustering Visualization

```python
from kfactors import plot_fuzzy_clusters

# Show membership strengths
plot_fuzzy_clusters(X_2d, memberships, centers=centers, show_all=True)
plt.show()
```

## Advanced Usage

### Custom Algorithm with Builder

```python
from kfactors import ClusteringBuilder

# Build custom algorithm by mixing components
algorithm = (ClusteringBuilder()
    .with_subspace_representation(dim=3)
    .with_hard_assignment()
    .with_pca_update()
    .with_kmeans_plusplus_init()
    .with_objective_convergence(tol=1e-4)
    .with_device('cuda')
    .build(n_clusters=5))

algorithm.fit(X)
```

### Data Preprocessing

```python
from kfactors.utils.validation import scale_data, handle_missing_values

# Scale data
X_scaled = scale_data(X, method='standard')  # or 'minmax', 'robust'

# Handle missing values
X_clean = handle_missing_values(X, method='mean')  # or 'drop_samples', 'median'
```

### Memory-Efficient Large Data

```python
from kfactors.utils.device import auto_select_device, estimate_memory_usage

# Estimate memory needs
mem_usage = estimate_memory_usage(
    n_samples=100000,
    n_features=1000,
    n_clusters=10,
    algorithm='kfactors'
)

# Auto-select best device based on data size
device = auto_select_device(
    data_size=X.nbytes,
    algorithm='kfactors'
)

model = KFactors(n_clusters=10, device=device)
```

## Implementation Architecture

The codebase follows a modular design that separates concerns:

```
kfactors/
├── algorithms/      # Main clustering algorithms
├── base/           # Core abstractions and interfaces
├── representations/# How clusters are modeled
├── assignments/    # How points are assigned to clusters
├── updates/        # How parameters are updated
├── initialization/ # How to initialize clusters
├── utils/          # Helper functions and metrics
└── visualization/  # Plotting utilities
```

This architecture allows for easy extension and experimentation with new clustering variants.

## Performance Characteristics

| Algorithm   | Time Complexity           | Space Complexity    | Best For                        |
|-------------|---------------------------|--------------------|---------------------------------|
| K-Means     | O(nkd)                   | O(nd + kd)         | Large data, spherical clusters  |
| K-Lines     | O(nkd + kd²)             | O(nd + kd)         | Data along 1D manifolds         |
| K-Subspaces | O(nkdR + kd²R)           | O(nd + kdR)        | Linear manifold data            |
| K-Factors   | O(nkdR² + kd²R)          | O(nd + ndR + kdR)  | Sequential subspace discovery   |
| GMM         | O(nkd² + kd³)            | O(nd + kd²)        | Flexible cluster shapes         |
| C-Factors   | O(nkdR + kd²R + kR³)     | O(nd + kdR)        | High-dim with low-rank structure|

where n=samples, k=clusters, d=features, R=subspace dimension

## Algorithm Selection Guide

- **K-Means**: Fast, simple, spherical clusters
- **K-Lines/K-Subspaces**: Data lies along linear subspaces
- **K-Factors**: Complex data with local subspace structure, need interpretable bases
- **GMM**: Need probabilistic assignments, non-spherical clusters
- **C-Factors**: High-dimensional data with low-rank structure
- **Fuzzy C-Means**: Overlapping clusters, soft boundaries

## Contributing

This is an active research project. Contributions are welcome for:
- Implementing planned features
- Adding new clustering variants
- Improving performance
- Adding visualization tools
- Extending applications

Please open an issue to discuss major changes before submitting a PR.

## Acknowledgements

This work builds on foundational contributions:
- MacQueen (1967): K-Means clustering
- Bradley & Mangasarian (2000): K-Planes/K-Subspaces
- Tipping & Bishop (1999): Probabilistic PCA and Mixtures thereof
- Ghahramani & Hinton (1996): Factor Analyzers

The K-Factors algorithm represents a novel synthesis of these ideas, introducing the sequential extraction mechanism with direction penalties.

## Citation

If you use K-Factors in your research, please cite:

```bibtex
@software{kfactors2024,
  title = {K-Factors: A Unified Framework for Clustering with Local Subspaces},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/kfactors}
}
```

## License

MIT License - see LICENSE file for details.

## Quick Start

### Basic Usage Pattern

All algorithms follow the sklearn pattern:

```python
import torch
from kfactors import KMeans, KFactors, GaussianMixture

# Generate sample data
X = torch.randn(1000, 20)

# Fit any algorithm
model = KMeans(n_clusters=5)  # or any other algorithm
model.fit(X)

# Get cluster assignments
labels = model.predict(X)

# Get cluster centers (available for all algorithms)
centers = model.cluster_centers_
```

### GPU Acceleration

```python
# Automatic GPU usage if available
model = KMeans(n_clusters=5, device='auto')

# Explicit device selection
model = KMeans(n_clusters=5, device='cuda:0')

# CPU only
model = KMeans(n_clusters=5, device='cpu')
```

## Core Algorithms

### 1. K-Means

Standard centroid-based clustering:

```python
from kfactors import KMeans

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',  # or 'random' or custom centers
    max_iter=100,
    tol=1e-4,
    verbose=1
)
kmeans.fit(X)

# Results
labels = kmeans.labels_
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_
```

### 2. K-Subspaces

Clusters data along r-dimensional affine subspaces:

```python
from kfactors import KSubspaces

# Fit 2D subspaces in 10D data
ksub = KSubspaces(
    n_clusters=4,
    n_components=2,  # subspace dimension
    max_iter=50
)
ksub.fit(X)

# Transform to local coordinates
X_local = ksub.transform(X)  # (n_samples, n_components)

# Reconstruct from local coordinates
X_reconstructed = ksub.inverse_transform(X_local)
```

### 3. K-Factors

Sequential subspace extraction with penalty mechanism:

```python
from kfactors import KFactors

kfactors = KFactors(
    n_clusters=4,
    n_components=3,  # dimensions per cluster
    penalty_weight=0.9,  # penalty for reusing directions
    max_iter=30
)
kfactors.fit(X)

# Inspect stage-wise progress
for stage in kfactors.stage_history_:
    print(f"Stage {stage['stage']}: {stage['iterations']} iterations")
```

### 4. Gaussian Mixture Model

Full probabilistic clustering with various covariance types:

```python
from kfactors import GaussianMixture

gmm = GaussianMixture(
    n_clusters=3,
    covariance_type='full',  # 'diagonal', 'spherical', 'tied'
    n_init=5,  # number of initializations
    max_iter=100
)
gmm.fit(X)

# Get soft assignments (probabilities)
probs = gmm.predict_proba(X)

# Generate new samples
X_new, y_new = gmm.sample(100)

# Model selection
bic = gmm.bic(X)
aic = gmm.aic(X)
```

### 5. C-Factors (Mixture of PPCA)

Soft clustering with low-rank Gaussian models:

```python
from kfactors import CFactors

cfactors = CFactors(
    n_clusters=4,
    n_components=3,  # latent dimensions
    max_iter=100
)
cfactors.fit(X)

# Transform to latent space
Z = cfactors.transform(X)

# Get soft assignments
responsibilities = cfactors.predict_proba(X)
```

### 6. Fuzzy C-Means

Soft clustering with fuzzy memberships:

```python
from kfactors import FuzzyCMeans

fcm = FuzzyCMeans(
    n_clusters=4,
    m=2.0,  # fuzziness parameter (m>1)
    max_iter=100
)
fcm.fit(X)

# Get fuzzy memberships
memberships = fcm.u_  # (n_samples, n_clusters)

# Evaluate fuzziness
fpc = fcm.partition_coefficient()  # closer to 1 = less fuzzy
fpe = fcm.partition_entropy()  # lower = less fuzzy
```

## Evaluation and Metrics

### Internal Metrics (no ground truth)

```python
from kfactors.utils.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Fit model
model = KMeans(n_clusters=5)
model.fit(X)
labels = model.labels_

# Evaluate
sil = silhouette_score(X, labels)  # higher is better [-1, 1]
db = davies_bouldin_score(X, labels)  # lower is better
ch = calinski_harabasz_score(X, labels)  # higher is better
```

### External Metrics (with ground truth)

```python
from kfactors.utils.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)

# Compare with true labels
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
homo, comp, v_score = v_measure_score(true_labels, pred_labels)
```

## Visualization

### Basic 2D/3D Visualization

```python
from kfactors import plot_clusters_2d, plot_clusters_3d
import matplotlib.pyplot as plt

# For 2D data
plot_clusters_2d(X_2d, labels, centers=centers)
plt.show()

# For 3D data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_clusters_3d(X_3d, labels, centers=centers, ax=ax)
plt.show()
```

### Fuzzy Clustering Visualization

```python
from kfactors import plot_fuzzy_clusters

# Show membership strengths
plot_fuzzy_clusters(X_2d, memberships, centers=centers, show_all=True)
plt.show()
```

### Decision Boundaries

```python
from kfactors import plot_cluster_boundaries

# Requires 2D data
plot_cluster_boundaries(X_2d, model, resolution=200)
plt.show()
```

## Advanced Usage

### Custom Algorithm with Builder

```python
from kfactors import ClusteringBuilder

# Build custom algorithm by mixing components
algorithm = (ClusteringBuilder()
    .with_subspace_representation(dim=3)
    .with_hard_assignment()
    .with_pca_update()
    .with_kmeans_plusplus_init()
    .with_objective_convergence(tol=1e-4)
    .with_device('cuda')
    .build(n_clusters=5))

algorithm.fit(X)
```

### Data Preprocessing

```python
from kfactors.utils.validation import scale_data, handle_missing_values

# Scale data
X_scaled = scale_data(X, method='standard')  # or 'minmax', 'robust'

# Handle missing values
X_clean = handle_missing_values(X, method='mean')  # or 'drop_samples', 'median'
```

### Memory-Efficient Large Data

```python
from kfactors.utils.device import auto_select_device, estimate_memory_usage

# Estimate memory needs
mem_usage = estimate_memory_usage(
    n_samples=100000,
    n_features=1000,
    n_clusters=10,
    algorithm='kfactors'
)

# Auto-select best device based on data size
device = auto_select_device(
    data_size=X.nbytes,
    algorithm='kfactors'
)

model = KFactors(n_clusters=10, device=device)
```

## Design Patterns

### 1. Consistent API

All algorithms implement:
- `fit(X)`: Train the model
- `predict(X)`: Get hard cluster assignments
- `fit_predict(X)`: Fit and return labels
- `cluster_centers_`: Access cluster centers

### 2. Soft Clustering

Algorithms with soft assignment additionally provide:
- `predict_proba(X)`: Get membership probabilities
- `score(X)`: Compute log-likelihood

### 3. Subspace Methods

Subspace-based algorithms (K-Subspaces, K-Factors, C-Factors) provide:
- `transform(X)`: Project to local coordinates
- `inverse_transform(X_transformed)`: Reconstruct original space

### 4. Component Architecture

The library uses modular components:
- **Representations**: How clusters are modeled (centroids, subspaces, Gaussians)
- **Assignments**: How points are assigned (hard, soft, fuzzy, penalized)
- **Updates**: How parameters are updated (mean, PCA, EM)
- **Initialization**: How to start (random, k-means++, spectral)

## Performance Tips

1. **Use GPU for large datasets**: Automatic with `device='auto'`
2. **Preprocessing**: Scale features for better convergence
3. **Multiple runs**: Use `n_init` for probabilistic algorithms
4. **Convergence**: Adjust `tol` and `max_iter` based on data size
5. **Memory**: Use `device.estimate_memory_usage()` for large data

## Algorithm Selection Guide

- **K-Means**: Fast, simple, spherical clusters
- **K-Lines/K-Subspaces**: Data lies along linear subspaces
- **K-Factors**: Complex data with local subspace structure
- **GMM**: Need probabilistic assignments, non-spherical clusters
- **C-Factors**: High-dimensional data with low-rank structure
- **Fuzzy C-Means**: Overlapping clusters, soft boundaries

## Citation

If you use K-Factors in your research, please cite:

```bibtex
@software{kfactors2024,
  title = {K-Factors: A Unified Framework for Clustering with Local Subspaces},
  author = {Alan Oursland},
  year = {2024},
  url = {https://github.com/alanoursland/kfactors}
}
```

