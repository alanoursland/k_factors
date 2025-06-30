"""
Random initialization strategy for clustering algorithms.

Selects random points from the dataset as initial cluster centers.
"""

from typing import List
import torch
from torch import Tensor

from ..base.interfaces import InitializationStrategy, ClusterRepresentation
from ..representations.centroid import CentroidRepresentation


class RandomInit(InitializationStrategy):
    """Random initialization by selecting points from the dataset.
    
    Selects n_clusters random points (without replacement) as initial centers.
    """
    
    def initialize(self, points: Tensor, n_clusters: int,
                  **kwargs) -> List[ClusterRepresentation]:
        """Initialize clusters with random points.
        
        Args:
            points: (n, d) data points
            n_clusters: Number of clusters
            
        Returns:
            List of initialized CentroidRepresentations
        """
        n_points, dimension = points.shape
        device = points.device
        
        if n_clusters > n_points:
            raise ValueError(f"Cannot create {n_clusters} clusters from {n_points} points")
            
        # Select random indices without replacement
        indices = torch.randperm(n_points, device=device)[:n_clusters]
        
        # Create representations
        representations = []
        for idx in indices:
            rep = CentroidRepresentation(dimension, device)
            rep.mean = points[idx].clone()
            representations.append(rep)
            
        return representations