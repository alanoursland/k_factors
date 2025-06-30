"""
Euclidean distance metric for clustering.

The most common distance metric, used in K-means and many other algorithms.
"""

import torch
from torch import Tensor

from ..base.interfaces import DistanceMetric, ClusterRepresentation


class EuclideanDistance(DistanceMetric):
    """Squared Euclidean distance metric.
    
    Computes ||x - μ||² where μ is the cluster center.
    """
    
    def __init__(self, squared: bool = True):
        """
        Args:
            squared: If True, return squared distances (default).
                    If False, return actual Euclidean distances.
        """
        self.squared = squared
        
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute Euclidean distances from points to cluster center.
        
        Args:
            points: (n, d) tensor of points
            representation: Cluster representation with 'mean' parameter
            
        Returns:
            (n,) tensor of distances
        """
        # Get cluster center
        params = representation.get_parameters()
        if 'mean' not in params:
            raise ValueError("Euclidean distance requires representation with 'mean' parameter")
            
        center = params['mean']
        
        # Compute squared distances
        diff = points - center.unsqueeze(0)
        squared_distances = torch.sum(diff * diff, dim=1)
        
        if self.squared:
            return squared_distances
        else:
            return torch.sqrt(squared_distances)
            
            
class WeightedEuclideanDistance(DistanceMetric):
    """Weighted Euclidean distance with feature weights.
    
    Computes sqrt(sum_i w_i * (x_i - μ_i)²) where w_i are feature weights.
    """
    
    def __init__(self, weights: Tensor, squared: bool = True):
        """
        Args:
            weights: (d,) tensor of feature weights
            squared: Whether to return squared distances
        """
        self.weights = weights
        self.squared = squared
        
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute weighted Euclidean distances.
        
        Args:
            points: (n, d) tensor of points
            representation: Cluster representation with 'mean'
            
        Returns:
            (n,) tensor of distances
        """
        params = representation.get_parameters()
        center = params['mean']
        
        # Ensure weights are on same device
        weights = self.weights.to(points.device)
        
        # Compute weighted squared distances
        diff = points - center.unsqueeze(0)
        weighted_sq_diff = weights.unsqueeze(0) * diff * diff
        squared_distances = torch.sum(weighted_sq_diff, dim=1)
        
        if self.squared:
            return squared_distances
        else:
            return torch.sqrt(squared_distances)