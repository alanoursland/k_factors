"""
Hard assignment strategy for clustering algorithms.

Assigns each point to its nearest cluster based on the distance metric.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import Tensor

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation


class HardAssignment(AssignmentStrategy):
    """Hard (discrete) assignment to nearest cluster.
    
    Each point is assigned to exactly one cluster based on minimum distance.
    Used in K-means, K-subspaces, and K-factors.
    """
    
    def __init__(self):
        """Initialize hard assignment strategy."""
        super().__init__()
        
    @property
    def is_soft(self) -> bool:
        """Hard assignments are not soft."""
        return False
        
    def compute_assignments(self, points: Tensor, 
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tensor:
        """Assign each point to nearest cluster.
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            **kwargs: Ignored for basic hard assignment
            
        Returns:
            (n,) tensor of cluster indices
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        
        # Compute distance matrix
        distances = torch.zeros(n_points, n_clusters, device=points.device)
        
        for k, representation in enumerate(representations):
            distances[:, k] = representation.distance_to_point(points)
            
        # Assign to nearest cluster (minimum distance)
        assignments = torch.argmin(distances, dim=1)
        
        return assignments
        
        
class HardAssignmentWithInfo(HardAssignment):
    """Hard assignment that also returns auxiliary information.
    
    Returns both assignments and a dictionary containing useful information
    like distances, second-best clusters, etc.
    """
    
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          return_distances: bool = True,
                          return_second_best: bool = False,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Assign points and return additional information.
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            return_distances: Whether to return distance matrix
            return_second_best: Whether to return second-best assignments
            
        Returns:
            assignments: (n,) cluster indices
            info: Dictionary with auxiliary information
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        
        # Compute distance matrix
        distances = torch.zeros(n_points, n_clusters, device=points.device)
        
        for k, representation in enumerate(representations):
            distances[:, k] = representation.distance_to_point(points)
            
        # Get assignments
        assignments = torch.argmin(distances, dim=1)
        
        # Build info dictionary
        info = {}
        
        if return_distances:
            info['distances'] = distances
            info['min_distances'] = torch.gather(distances, 1, assignments.unsqueeze(1)).squeeze(1)
            
        if return_second_best:
            # Mask out the best cluster and find second best
            mask = torch.ones_like(distances, dtype=torch.bool)
            mask.scatter_(1, assignments.unsqueeze(1), False)
            masked_distances = torch.where(mask, distances, float('inf'))
            
            second_best = torch.argmin(masked_distances, dim=1)
            second_best_distances = torch.gather(masked_distances, 1, second_best.unsqueeze(1)).squeeze(1)
            
            info['second_best'] = second_best
            info['second_best_distances'] = second_best_distances
            info['assignment_confidence'] = (second_best_distances - info['min_distances']) / (second_best_distances + 1e-10)
            
        return assignments, info