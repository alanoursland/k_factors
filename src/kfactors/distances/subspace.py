"""
Subspace distance metrics for clustering.

Used in K-subspaces, K-lines, and related algorithms that model
clusters as linear subspaces.
"""

import torch
from torch import Tensor

from ..base.interfaces import DistanceMetric, ClusterRepresentation


class OrthogonalDistance(DistanceMetric):
    """Orthogonal distance from points to subspace.
    
    Computes the squared distance from each point to its orthogonal
    projection onto the subspace.
    """
    
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute orthogonal distances to subspace.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean' and 'basis' parameters
            
        Returns:
            (n,) tensor of squared orthogonal distances
        """
        params = representation.get_parameters()
        
        if 'mean' not in params:
            raise ValueError("Subspace distance requires 'mean' parameter")
        if 'basis' not in params:
            raise ValueError("Subspace distance requires 'basis' parameter")
            
        mean = params['mean']
        basis = params['basis']  # (d, r) orthonormal basis
        
        # Center points
        centered = points - mean.unsqueeze(0)
        
        if basis.shape[1] == 0:
            # Empty subspace - distance is just distance to mean
            return torch.sum(centered * centered, dim=1)
            
        # Project onto subspace: P = B B^T
        # Projection of x is: B B^T x
        # Distance is: ||x - B B^T x||² = ||x||² - ||B^T x||²
        
        # Compute B^T x (coefficients in subspace)
        coeffs = torch.matmul(centered, basis)  # (n, r)
        
        # ||B^T x||²
        proj_norm_sq = torch.sum(coeffs * coeffs, dim=1)
        
        # ||x||²
        centered_norm_sq = torch.sum(centered * centered, dim=1)
        
        # Orthogonal distance squared
        distances = centered_norm_sq - proj_norm_sq
        
        # Numerical safety - distances should be non-negative
        distances = torch.clamp(distances, min=0.0)
        
        return distances
        
        
class SubspaceAngleDistance(DistanceMetric):
    """Distance based on angle between point and subspace.
    
    Computes sin²(θ) where θ is the angle between the point
    and its projection onto the subspace.
    """
    
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute angle-based distances.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean' and 'basis'
            
        Returns:
            (n,) tensor of sin²(angle) values
        """
        params = representation.get_parameters()
        mean = params['mean']
        basis = params['basis']
        
        # Center points
        centered = points - mean.unsqueeze(0)
        
        # Compute norms
        point_norms = torch.norm(centered, dim=1, keepdim=True)
        
        # Avoid division by zero
        point_norms = torch.clamp(point_norms, min=1e-8)
        normalized = centered / point_norms
        
        if basis.shape[1] == 0:
            # No subspace - angle is 90 degrees
            return torch.ones(points.shape[0], device=points.device)
            
        # Project normalized points
        coeffs = torch.matmul(normalized, basis)
        
        # cos²(θ) = ||projection||²
        cos_sq = torch.sum(coeffs * coeffs, dim=1)
        
        # sin²(θ) = 1 - cos²(θ)
        sin_sq = 1.0 - cos_sq
        
        return sin_sq
        

class GrassmannDistance(DistanceMetric):
    """Grassmann distance between subspaces.
    
    Used when comparing subspaces themselves rather than
    points to subspaces. Useful for subspace clustering
    where each data point is itself a subspace.
    """
    
    def __init__(self, metric: str = 'projection'):
        """
        Args:
            metric: Type of Grassmann distance
                   'projection': Based on projection operator difference
                   'principal_angles': Based on principal angles
        """
        self.metric = metric
        
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute Grassmann distances.
        
        Args:
            points: (n, d, r1) tensor where each point is a subspace basis
            representation: Cluster with 'basis' of shape (d, r2)
            
        Returns:
            (n,) tensor of distances
        """
        if points.dim() != 3:
            raise ValueError("Grassmann distance expects points to be subspace bases "
                           f"of shape (n, d, r), got shape {points.shape}")
                           
        params = representation.get_parameters()
        if 'basis' not in params:
            raise ValueError("Grassmann distance requires 'basis' parameter")
            
        basis2 = params['basis']  # (d, r2)
        n, d, r1 = points.shape
        r2 = basis2.shape[1]
        
        if self.metric == 'projection':
            # Distance based on ||P1 - P2||_F where P = BB^T
            distances = torch.zeros(n, device=points.device)
            
            for i in range(n):
                basis1 = points[i]  # (d, r1)
                
                # Compute projection operators
                P1 = torch.matmul(basis1, basis1.t())
                P2 = torch.matmul(basis2, basis2.t())
                
                # Frobenius norm of difference
                diff = P1 - P2
                distances[i] = torch.norm(diff, p='fro') ** 2
                
        elif self.metric == 'principal_angles':
            # Distance based on principal angles
            distances = torch.zeros(n, device=points.device)
            
            for i in range(n):
                basis1 = points[i]  # (d, r1)
                
                # Compute B1^T B2
                M = torch.matmul(basis1.t(), basis2)  # (r1, r2)
                
                # SVD gives cosines of principal angles
                _, S, _ = torch.linalg.svd(M)
                
                # Clamp to [0, 1] for numerical safety
                S = torch.clamp(S, min=0.0, max=1.0)
                
                # Principal angles
                angles = torch.acos(S)
                
                # Distance is sum of squared angles
                distances[i] = torch.sum(angles ** 2)
                
        else:
            raise ValueError(f"Unknown Grassmann metric: {self.metric}")
            
        return distances