"""
Centroid representation for K-means clustering.

The simplest cluster representation - just a mean point in space.
"""

from typing import Dict, Optional
import torch
from torch import Tensor

from .base_representation import BaseRepresentation


class CentroidRepresentation(BaseRepresentation):
    """Cluster represented by a single centroid point.
    
    Used in K-means and as a component of more complex representations.
    """
    
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """Compute Euclidean distance from points to centroid.
        
        Args:
            points: (n, d) tensor of data points
            indices: Ignored for centroid representation
            
        Returns:
            (n,) tensor of squared Euclidean distances
        """
        self._check_points_shape(points)
        
        # Squared Euclidean distance: ||x - μ||²
        diff = points - self._mean.unsqueeze(0)
        distances = torch.sum(diff * diff, dim=1)
        
        return distances
        
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None, 
                          **kwargs) -> None:
        """Update centroid as mean of assigned points.
        
        Args:
            points: (n, d) tensor of assigned points
            weights: Optional (n,) tensor of weights for soft assignments
        """
        self._check_points_shape(points)
        
        if len(points) == 0:
            # No points assigned - keep current mean
            return
            
        if weights is None:
            # Hard assignment - simple mean
            self._mean = points.mean(dim=0)
        else:
            # Soft assignment - weighted mean
            assert weights.shape == (points.shape[0],)
            weights = weights.to(self._device)
            
            # Normalize weights
            total_weight = weights.sum()
            if total_weight > 0:
                normalized_weights = weights / total_weight
                self._mean = torch.sum(
                    points * normalized_weights.unsqueeze(1), 
                    dim=0
                )
                
    def get_parameters(self) -> Dict[str, Tensor]:
        """Return parameters defining this centroid."""
        return {'mean': self._mean.clone()}
        
    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set centroid parameters."""
        if 'mean' in params:
            self.mean = params['mean']
            
    def __repr__(self) -> str:
        return f"CentroidRepresentation(dimension={self._dimension}, mean_norm={self._mean.norm():.3f})"