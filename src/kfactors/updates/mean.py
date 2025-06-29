"""
Mean update strategy for centroid-based clustering.
"""

from typing import Optional
import torch
from torch import Tensor

from ..base.interfaces import ParameterUpdater, ClusterRepresentation


class MeanUpdater(ParameterUpdater):
    """Updates cluster representation by computing mean of assigned points."""
    
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignments: Optional[Tensor] = None,
               **kwargs) -> None:
        """Update cluster mean.
        
        Args:
            representation: Cluster representation to update
            points: Points assigned to this cluster (already filtered)
            assignments: Weights for soft assignment or None for hard
            **kwargs: Ignored
        """
        # Just delegate to the representation's update method
        representation.update_from_points(points, weights=assignments, **kwargs)