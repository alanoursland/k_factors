"""
Ranked assignment strategies that order features by distance to points.

These strategies compute distance-based rankings rather than traditional
hard or soft cluster assignments. Used for algorithms like K-Factors
where points need to be ranked against all features.
"""

from typing import List, Tuple, Dict, Any
import torch
from torch import Tensor

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation
from ..representations.centroid import CentroidRepresentation
from ..representations.eigenfactor import EigenFactorRepresentation


class CentroidRanker(AssignmentStrategy):
    """Ranks centroid features by Euclidean distance to points.
    
    For each point, computes distances to all centroids and returns
    feature indices sorted by distance (closest first).
    """
    
    @property
    def is_soft(self) -> bool:
        """Centroid ranking produces ranked assignments."""
        return False
    
    @property
    def is_ranked(self) -> bool:
        """This strategy produces ranked assignments."""
        return True
    
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute centroid rankings by Euclidean distance.
        
        Uses the efficient squared distance formula: ||x-y||^2 = ||x||^2 - 2xy + ||y||^2
        
        Args:
            points: (n, d) tensor of data points
            representations: List of CentroidRepresentation objects
            **kwargs: Additional parameters (ignored)
            
        Returns:
            rankings: (n, K) tensor of feature indices ordered by distance
            aux_info: {'distances': (n, K) corresponding squared distances}
        """
        n_points = points.shape[0]
        n_centroids = len(representations)
        device = points.device
        
        # Verify all representations are centroids
        for rep in representations:
            if not isinstance(rep, CentroidRepresentation):
                raise TypeError(f"CentroidRanker requires CentroidRepresentation, "
                              f"got {type(rep)}")
        
        # Extract all centroids into a single tensor for efficient computation
        centroids = torch.stack([rep.mean for rep in representations])  # (K, d)
        
        # Efficient squared distance computation: ||x-y||^2 = ||x||^2 - 2xy + ||y||^2
        # points: (n, d), centroids: (K, d)
        x_norm_sq = torch.sum(points * points, dim=1, keepdim=True)  # (n, 1)
        y_norm_sq = torch.sum(centroids * centroids, dim=1)          # (K,)
        xy = torch.matmul(points, centroids.t())                     # (n, K)
        
        # Broadcast and compute squared distances
        distances = x_norm_sq - 2 * xy + y_norm_sq.unsqueeze(0)     # (n, K)
        
        # Sort by distance (ascending - closest first)
        rankings = torch.argsort(distances, dim=1)
        sorted_distances = torch.gather(distances, 1, rankings)
        
        aux_info = {
            'distances': sorted_distances,
            'raw_distances': distances
        }
        
        return rankings, aux_info


class HyperplaneRanker(AssignmentStrategy):
    """Ranks hyperplane features by distance to points.
    
    For flat EigenFactorRepresentation, computes distances from points
    to each hyperplane and returns feature indices sorted by distance.
    """
    
    @property
    def is_soft(self) -> bool:
        """Hyperplane ranking produces ranked assignments."""
        return False
    
    @property
    def is_ranked(self) -> bool:
        """This strategy produces ranked assignments.""" 
        return True
    
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute hyperplane rankings by Mahalanobis distance.
        
        Uses Mahalanobis distance: D = w_k^T(x - mu_k) where w_k = v_k/sqrt(lambda_k)
        The stored vectors already incorporate the scaling, so we compute y = w_k^T(x - mu_k)
        
        Args:
            points: (n, d) tensor of data points  
            representations: Should contain single EigenFactorRepresentation
            **kwargs: Additional parameters (ignored)
            
        Returns:
            rankings: (n, K) tensor of feature indices ordered by distance
            aux_info: {'distances': (n, K) squared Mahalanobis distances}
        """
        if len(representations) != 1:
            raise ValueError("HyperplaneRanker expects exactly one EigenFactorRepresentation")
        
        rep = representations[0]
        if not isinstance(rep, EigenFactorRepresentation):
            raise TypeError(f"HyperplaneRanker requires EigenFactorRepresentation, "
                          f"got {type(rep)}")
        
        n_points = points.shape[0]
        n_factors = rep.n_factors
        device = points.device
        
        # Efficient computation: y = W(x - mu) where W is (K, d) and points is (n, d)
        # Alternative form: y = Wx - W*mu for potentially better numerical properties
        
        # Method 1: Direct computation y = W(x - mu)
        # Expand for broadcasting: (n, 1, d) - (1, K, d) = (n, K, d)
        x_expanded = points.unsqueeze(1)               # (n, 1, d)
        mu_expanded = rep.means.unsqueeze(0)           # (1, K, d)
        centered = x_expanded - mu_expanded            # (n, K, d)
        
        # Apply vectors: w_k^T(x - mu_k) for all k
        w_expanded = rep.vectors.unsqueeze(0)          # (1, K, d)
        projections = (centered * w_expanded).sum(dim=2)  # (n, K)
        
        # Squared Mahalanobis distances (no additional normalization needed 
        # since vectors already incorporate the 1/sqrt(lambda) scaling)
        distances = projections * projections          # (n, K)
        
        # Sort by distance (ascending - closest hyperplane first)
        rankings = torch.argsort(distances, dim=1)
        sorted_distances = torch.gather(distances, 1, rankings)
        
        aux_info = {
            'distances': sorted_distances,
            'raw_distances': distances,
            'projections': projections
        }
        
        return rankings, aux_info


class NearestFeatureRanker(AssignmentStrategy):
    """Generic ranker that works with any representation type.
    
    Automatically dispatches to appropriate ranking strategy based on
    representation type. Useful for mixed representation types.
    """
    
    def __init__(self):
        self._centroid_ranker = CentroidRanker()
        self._hyperplane_ranker = HyperplaneRanker()
    
    @property
    def is_soft(self) -> bool:
        """Generic ranking produces ranked assignments."""
        return False
        
    @property
    def is_ranked(self) -> bool:
        """This strategy produces ranked assignments."""
        return True
    
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Dispatch to appropriate ranker based on representation type.
        
        Args:
            points: (n, d) tensor of data points
            representations: List of cluster representations
            **kwargs: Additional parameters
            
        Returns:
            rankings: (n, K) tensor of feature indices ordered by distance
            aux_info: Dictionary with ranking information
        """
        if not representations:
            raise ValueError("No representations provided")
        
        # Check if all representations are of the same type
        rep_types = {type(rep) for rep in representations}
        
        if len(rep_types) == 1:
            rep_type = next(iter(rep_types))
            
            if rep_type == CentroidRepresentation:
                return self._centroid_ranker.compute_assignments(
                    points, representations, **kwargs
                )
            elif rep_type == EigenFactorRepresentation:
                return self._hyperplane_ranker.compute_assignments(
                    points, representations, **kwargs
                )
        
        # Fallback: compute distances generically
        return self._compute_generic_rankings(points, representations, **kwargs)
    
    def _compute_generic_rankings(self, points: Tensor,
                                representations: List[ClusterRepresentation],
                                **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Generic distance-based ranking for mixed representation types."""
        n_points = points.shape[0]
        n_features = len(representations)
        device = points.device
        
        # Compute distances using each representation's distance method
        distances = torch.zeros(n_points, n_features, device=device)
        
        for k, rep in enumerate(representations):
            distances[:, k] = rep.distance_to_point(points)
        
        # Sort by distance (ascending)
        rankings = torch.argsort(distances, dim=1)
        sorted_distances = torch.gather(distances, 1, rankings)
        
        aux_info = {
            'distances': sorted_distances,
            'raw_distances': distances,
            'representation_types': [type(rep).__name__ for rep in representations]
        }
        
        return rankings, aux_info