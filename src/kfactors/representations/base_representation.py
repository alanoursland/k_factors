"""
Base representation class with common functionality for all cluster representations.
"""

from typing import Dict, Optional
import torch
from torch import Tensor

from ..base.interfaces import ClusterRepresentation


class BaseRepresentation(ClusterRepresentation):
    """Base class providing common functionality for cluster representations."""
    
    def __init__(self, dimension: int, device: torch.device):
        """
        Args:
            dimension: Ambient dimension d of the data
            device: Torch device for tensor allocation
        """
        self._dimension = dimension
        self._device = device
        self._mean = torch.zeros(dimension, device=device)
        
    @property
    def dimension(self) -> int:
        """Ambient dimension of the data."""
        return self._dimension
        
    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self._device
        
    @property
    def mean(self) -> Tensor:
        """Cluster mean/centroid."""
        return self._mean
        
    @mean.setter
    def mean(self, value: Tensor):
        """Set cluster mean."""
        assert value.shape == (self._dimension,)
        self._mean = value.to(self._device)
        
    def to(self, device: torch.device) -> 'BaseRepresentation':
        """Move representation to specified device."""
        # Create new instance on target device
        new_repr = self.__class__(self._dimension, device)
        
        # Copy all tensor attributes
        params = self.get_parameters()
        new_params = {k: v.to(device) for k, v in params.items()}
        new_repr.set_parameters(new_params)
        
        return new_repr
        
    def _check_points_shape(self, points: Tensor):
        """Validate shape of input points."""
        if points.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {points.dim()}D")
        if points.shape[1] != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {points.shape[1]}")