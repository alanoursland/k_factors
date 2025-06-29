"""
Demo of K-Factors clustering algorithm.

This example shows how to:
1. Generate synthetic data with local subspace structure
2. Apply K-Factors clustering
3. Visualize results and compare with K-means
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
import sys
sys.path.append('..')

from kfactors.algorithms.kfactors import KFactors
from kfactors.algorithms.kmeans import KMeans


def generate_subspace_data(n_points_per_cluster=200, n_clusters=3, 
                          ambient_dim=10, subspace_dim=2, noise_level=0.1):
    """Generate synthetic data with local subspace structure.
    
    Each cluster lies approximately on a different 2D subspace in 10D space.
    """
    torch.manual_seed(42)
    
    data_list = []
    true_labels = []
    
    for k in range(n_clusters):
        # Random orthonormal basis for this cluster's subspace
        basis = torch.randn(ambient_dim, subspace_dim)
        basis, _ = torch.linalg.qr(basis)
        
        # Random mean for this cluster
        mean = torch.randn(ambient_dim) * 3
        
        # Generate points in subspace
        # Coefficients in subspace (make them somewhat spread out)
        coeffs = torch.randn(n_points_per_cluster, subspace_dim) * 2
        
        # Project to amb