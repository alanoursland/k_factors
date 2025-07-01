"""
K-Factors: A unified framework for clustering with local subspaces.

This package implements the K-Factors family of clustering algorithms, including:
- K-means
- K-lines/K-subspaces  
- K-factors (sequential PPCA)
- C-factors (soft Mixture of PPCA)
- Fuzzy C-means
- Gaussian Mixture Models

Example usage:
    >>> import torch
    >>> from kfactors import KMeans
    >>> 
    >>> # Generate sample data
    >>> X = torch.randn(1000, 10)
    >>> 
    >>> # Fit K-means
    >>> kmeans = KMeans(n_clusters=5, verbose=1)
    >>> kmeans.fit(X)
    >>> 
    >>> # Get cluster assignments
    >>> labels = kmeans.predict(X)
"""

__version__ = '0.1.0'

# Import main algorithms
from .algorithms.kmeans import KMeans
from .algorithms.kfactors import KFactors
from .algorithms.klines import KLines
from .algorithms.ksubspaces import KSubspaces
from .algorithms.gmm import GaussianMixture
from .algorithms.cfactors import CFactors
from .algorithms.fuzzy_cmeans import FuzzyCMeans, PossibilisticCMeans, FuzzyPossibilisticCMeans
from .algorithms.builder import ClusteringBuilder, create_kmeans, create_ksubspaces

# Import visualization
from .visualization import (
    plot_clusters_2d,
    plot_clusters_3d,
    plot_fuzzy_clusters,
    plot_cluster_boundaries,
    plot_gaussian_ellipses
)

# Convenience imports
from .base import (
    ClusterState,
    AssignmentMatrix,
    DirectionTracker
)

__all__ = [
    # Algorithms
    'KMeans',
    'KFactors',
    'KLines',
    'KSubspaces',
    'GaussianMixture',
    'CFactors',
    'FuzzyCMeans',
    'PossibilisticCMeans',
    'FuzzyPossibilisticCMeans',
    
    # Builder
    'ClusteringBuilder',
    'create_kmeans',
    'create_ksubspaces',
    
    # Core data structures  
    'ClusterState',
    'AssignmentMatrix',
    'DirectionTracker',
    
    # Visualization
    'plot_clusters_2d',
    'plot_clusters_3d',
    'plot_fuzzy_clusters',
    'plot_cluster_boundaries',
    'plot_gaussian_ellipses',
    
    # Version
    '__version__'
]

# TODO: Add these as they're implemented:
# from .algorithms.klines import KLines
# from .algorithms.ksubspaces import KSubspaces  
# from .algorithms.kfactors import KFactors
# from .algorithms.cfactors import CFactors
# from .algorithms.fuzzy_cmeans import FuzzyCMeans
# from .algorithms.gmm import GaussianMixture