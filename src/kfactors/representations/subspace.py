"""
Subspace representation for K-Subspaces/K-Lines clustering.

Represents clusters as affine subspaces (point + linear subspace).
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor

from .base_representation import BaseRepresentation
from ..utils.linalg import orthogonalize_basis


class SubspaceRepresentation(BaseRepresentation):
    """Cluster represented by an affine subspace.
    
    The subspace is defined by:
    - A mean point μ
    - An orthonormal basis {v₁, ..., vᵣ} spanning an r-dimensional subspace
    
    Points are assigned based on orthogonal distance to the subspace.
    """
    
    def __init__(self, dimension: int, subspace_dim: int, device: torch.device):
        """
        Args:
            dimension: Ambient dimension d
            subspace_dim: Subspace dimension r (r ≤ d)
            device: Torch device
        """
        super().__init__(dimension, device)
        
        if subspace_dim > dimension:
            raise ValueError(f"Subspace dimension {subspace_dim} cannot exceed "
                           f"ambient dimension {dimension}")
                           
        self.subspace_dim = subspace_dim
        
        # Initialize with random orthonormal basis
        if subspace_dim > 0:
            random_matrix = torch.randn(dimension, subspace_dim, device=device)
            self._basis, _ = torch.linalg.qr(random_matrix)
        else:
            self._basis = torch.empty(dimension, 0, device=device)
            
    @property
    def basis(self) -> Tensor:
        """Orthonormal basis vectors (d, r)."""
        return self._basis
        
    @basis.setter
    def basis(self, value: Tensor):
        """Set basis vectors (will be orthonormalized)."""
        assert value.shape == (self._dimension, self.subspace_dim)
        self._basis = orthogonalize_basis(value.to(self._device))
        
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """Compute orthogonal distance from points to subspace.
        
        Distance = ||x - μ - VVᵀ(x - μ)||² where V is the basis matrix.
        
        Args:
            points: (n, d) tensor of data points
            indices: Ignored for basic subspace representation
            
        Returns:
            (n,) tensor of squared orthogonal distances
        """
        self._check_points_shape(points)
        
        # Center points
        centered = points - self._mean.unsqueeze(0)  # (n, d)
        
        if self.subspace_dim == 0:
            # No subspace - distance is just distance to mean
            return torch.sum(centered * centered, dim=1)
            
        # Project onto subspace: VVᵀ(x - μ)
        # First: Vᵀ(x - μ) gives coefficients
        coeffs = torch.matmul(centered, self._basis)  # (n, r)
        
        # Then: V * coeffs gives projection
        projections = torch.matmul(coeffs, self._basis.t())  # (n, d)
        
        # Residual after projection
        residuals = centered - projections  # (n, d)
        
        # Squared norm of residuals
        distances = torch.sum(residuals * residuals, dim=1)
        
        return distances
        
    def project_points(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Project points onto the subspace.
        
        Args:
            points: (n, d) points to project
            
        Returns:
            coefficients: (n, r) coordinates in the subspace
            projections: (n, d) projected points in ambient space
        """
        self._check_points_shape(points)
        
        centered = points - self._mean.unsqueeze(0)
        
        if self.subspace_dim == 0:
            # No subspace - projection is just the mean
            coeffs = torch.empty(points.shape[0], 0, device=self._device)
            projections = self._mean.unsqueeze(0).expand_as(points)
            return coeffs, projections
            
        # Get coefficients in subspace basis
        coeffs = torch.matmul(centered, self._basis)
        
        # Reconstruct in ambient space
        projections = self._mean.unsqueeze(0) + torch.matmul(coeffs, self._basis.t())
        
        return coeffs, projections
        
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None,
                          **kwargs) -> None:
        """Update subspace using PCA on assigned points.
        
        Args:
            points: (n, d) assigned points
            weights: Optional (n,) weights for soft assignments
        """
        self._check_points_shape(points)
        
        if len(points) == 0:
            return
            
        # Update mean
        if weights is None:
            self._mean = points.mean(dim=0)
            centered = points - self._mean.unsqueeze(0)
        else:
            assert weights.shape == (points.shape[0],)
            weights = weights.to(self._device)
            total_weight = weights.sum()
            
            if total_weight <= 0:
                return
                
            # Weighted mean
            normalized_weights = weights / total_weight
            self._mean = torch.sum(points * normalized_weights.unsqueeze(1), dim=0)
            centered = points - self._mean.unsqueeze(0)
            
            # Apply sqrt of weights for weighted PCA
            centered = centered * torch.sqrt(normalized_weights).unsqueeze(1)
            
        if self.subspace_dim == 0:
            return  # No subspace to update
            
        # Compute SVD of centered points
        try:
            # Use low-rank SVD if possible
            if centered.shape[0] < centered.shape[1] and self.subspace_dim < centered.shape[0]:
                U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                V = Vt.t()
            else:
                # Full SVD
                U, S, Vt = torch.linalg.svd(centered)
                V = Vt.t()
                
            # Take top r components
            self._basis = V[:, :self.subspace_dim]
            
        except Exception as e:
            # SVD can fail for degenerate cases
            import warnings
            warnings.warn(f"SVD failed in subspace update: {e}. Keeping previous basis.")
            
    def get_parameters(self) -> Dict[str, Tensor]:
        """Get all parameters."""
        return {
            'mean': self._mean.clone(),
            'basis': self._basis.clone()
        }
        
    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set parameters."""
        if 'mean' in params:
            self.mean = params['mean']
        if 'basis' in params:
            self.basis = params['basis']
            
    def __repr__(self) -> str:
        return (f"SubspaceRepresentation(dimension={self._dimension}, "
                f"subspace_dim={self.subspace_dim})")