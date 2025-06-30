"""
Mahalanobis distance metric for clustering.

Used in Gaussian-based clustering methods (GMM, C-Factors) where
clusters have different covariance structures.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import warnings

from ..base.interfaces import DistanceMetric, ClusterRepresentation
from ..utils.linalg import solve_linear_system, safe_eigh


class MahalanobisDistance(DistanceMetric):
    """Mahalanobis distance using cluster covariance.
    
    Computes (x - μ)^T Σ^(-1) (x - μ) where Σ is the covariance matrix.
    """
    
    def __init__(self, regularization: float = 1e-6):
        """
        Args:
            regularization: Small value added to diagonal for numerical stability
        """
        self.regularization = regularization
        
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute Mahalanobis distances.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean' and either 'covariance' or 'precision'
            
        Returns:
            (n,) tensor of squared Mahalanobis distances
        """
        params = representation.get_parameters()
        
        if 'mean' not in params:
            raise ValueError("Mahalanobis distance requires 'mean' parameter")
            
        mean = params['mean']
        centered = points - mean.unsqueeze(0)
        
        # Check for precision matrix first (more efficient if available)
        if 'precision' in params:
            precision = params['precision']
            
            # Compute x^T P x efficiently
            # First compute P x
            Px = torch.matmul(centered, precision.t())  # (n, d)
            # Then x^T (P x)
            distances = torch.sum(centered * Px, dim=1)
            
        elif 'covariance' in params:
            covariance = params['covariance']
            
            # Add regularization for numerical stability
            if self.regularization > 0:
                eye = torch.eye(covariance.shape[0], device=covariance.device)
                cov_reg = covariance + self.regularization * eye
            else:
                cov_reg = covariance
                
            # Solve Σ x = centered^T for x
            # This is more stable than computing Σ^(-1) explicitly
            try:
                # Cholesky decomposition for positive definite matrices
                L = torch.linalg.cholesky(cov_reg)
                
                # Solve L y = centered^T
                y = torch.linalg.solve_triangular(L, centered.t(), upper=False)
                
                # Distance is ||y||²
                distances = torch.sum(y * y, dim=0)
                
            except Exception as e:
                warnings.warn(f"Cholesky failed, using general solver: {e}")
                
                # Fallback to general linear solver
                # Solve for each point (less efficient but more stable)
                distances = torch.zeros(points.shape[0], device=points.device)
                for i in range(points.shape[0]):
                    sol = solve_linear_system(cov_reg, centered[i], method='lstsq')
                    distances[i] = torch.dot(centered[i], sol)
                    
        else:
            raise ValueError("Mahalanobis distance requires either 'covariance' or 'precision'")
            
        # Distances should be non-negative
        distances = torch.clamp(distances, min=0.0)
        
        return distances
        

class PrincipalAxisMahalanobisDistance(DistanceMetric):
    """Mahalanobis distance decomposed along principal axes.
    
    Computes the Mahalanobis distance as a sum of squared standardized
    distances along each eigenvector direction:
    
    d² = Σ_i (λ_i^(-1/2) v_i^T (x - μ))²
    
    where λ_i are eigenvalues and v_i are eigenvectors of the covariance.
    
    This decomposition is useful for:
    - Understanding which principal directions contribute most to distance
    - Efficient computation when only top eigenvalues are needed
    - Robust handling of near-singular covariances
    - Adaptive rank selection based on explained variance
    """
    
    def __init__(self, 
                 n_components: Optional[int] = None,
                 min_eigenvalue: float = 1e-6,
                 variance_fraction: Optional[float] = None):
        """
        Args:
            n_components: Number of principal components to use.
                         If None, uses all with eigenvalue > min_eigenvalue
            min_eigenvalue: Minimum eigenvalue to consider (filters noise)
            variance_fraction: If specified (e.g., 0.95), keep enough components
                             to explain this fraction of total variance.
                             Overrides n_components if both are specified.
        """
        self.n_components = n_components
        self.min_eigenvalue = min_eigenvalue
        self.variance_fraction = variance_fraction
        
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute Mahalanobis distance using eigendecomposition.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean' and either 'covariance' or 
                          ('eigenvalues', 'eigenvectors')
            
        Returns:
            (n,) tensor of squared Mahalanobis distances
        """
        params = representation.get_parameters()
        
        if 'mean' not in params:
            raise ValueError("Distance requires 'mean' parameter")
            
        mean = params['mean']
        centered = points - mean.unsqueeze(0)  # (n, d)
        
        # Get eigendecomposition
        if 'eigenvalues' in params and 'eigenvectors' in params:
            eigenvalues = params['eigenvalues']
            eigenvectors = params['eigenvectors']
        elif 'covariance' in params:
            # Compute eigendecomposition
            eigenvalues, eigenvectors = safe_eigh(params['covariance'])
            # Sort in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            raise ValueError("Distance requires either 'covariance' or "
                           "('eigenvalues', 'eigenvectors')")
            
        # Filter by minimum eigenvalue first
        valid_mask = eigenvalues > self.min_eigenvalue
        eigenvalues = eigenvalues[valid_mask]
        eigenvectors = eigenvectors[:, valid_mask]
        
        # Select components based on variance fraction if specified
        if self.variance_fraction is not None and len(eigenvalues) > 0:
            # Compute cumulative variance explained
            total_variance = eigenvalues.sum()
            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / total_variance
            
            # Find number of components needed
            n_comp = torch.searchsorted(cumulative_variance, self.variance_fraction).item() + 1
            n_comp = min(n_comp, len(eigenvalues))
            
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]
            
        elif self.n_components is not None:
            # Use fixed number of components
            n_comp = min(self.n_components, len(eigenvalues))
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]
            
        # Project onto eigenvectors: (n, k) = (n, d) @ (d, k)
        projections = torch.matmul(centered, eigenvectors)
        
        # Scale by inverse square root of eigenvalues
        # This gives the standardized distance along each principal axis
        scaled_projections = projections / torch.sqrt(eigenvalues).unsqueeze(0)
        
        # Sum of squared standardized distances
        distances = torch.sum(scaled_projections ** 2, dim=1)
        
        return distances
        
    def compute_per_component(self, points: Tensor, 
                            representation: ClusterRepresentation,
                            **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute distance contribution from each principal component.
        
        Returns the squared standardized distance along each eigenvector,
        useful for analysis and visualization.
        
        Args:
            points: (n, d) tensor of points
            representation: Cluster representation
            
        Returns:
            distances: (n, k) tensor where k is number of components used
            eigenvalues: (k,) tensor of eigenvalues used
        """
        params = representation.get_parameters()
        mean = params['mean']
        centered = points - mean.unsqueeze(0)
        
        # Get eigendecomposition
        if 'eigenvalues' in params and 'eigenvectors' in params:
            eigenvalues = params['eigenvalues']
            eigenvectors = params['eigenvectors']
        elif 'covariance' in params:
            eigenvalues, eigenvectors = safe_eigh(params['covariance'])
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            raise ValueError("Need covariance or eigendecomposition")
            
        # Filter by minimum eigenvalue
        valid_mask = eigenvalues > self.min_eigenvalue
        eigenvalues = eigenvalues[valid_mask]
        eigenvectors = eigenvectors[:, valid_mask]
        
        # Select based on variance fraction
        if self.variance_fraction is not None and len(eigenvalues) > 0:
            total_variance = eigenvalues.sum()
            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / total_variance
            n_comp = torch.searchsorted(cumulative_variance, self.variance_fraction).item() + 1
            n_comp = min(n_comp, len(eigenvalues))
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]
        elif self.n_components is not None:
            n_comp = min(self.n_components, len(eigenvalues))
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]
            
        # Project and scale
        projections = torch.matmul(centered, eigenvectors)
        scaled_projections = projections / torch.sqrt(eigenvalues).unsqueeze(0)
        
        # Return squared distances per component and eigenvalues
        return scaled_projections ** 2, eigenvalues
        
    def get_effective_rank(self, representation: ClusterRepresentation) -> int:
        """Get the effective rank based on current settings.
        
        Args:
            representation: Cluster representation with covariance
            
        Returns:
            Number of components that would be used
        """
        params = representation.get_parameters()
        
        if 'eigenvalues' in params:
            eigenvalues = params['eigenvalues']
        elif 'covariance' in params:
            eigenvalues, _ = safe_eigh(params['covariance'])
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        else:
            raise ValueError("Need eigenvalues or covariance")
            
        # Apply minimum threshold
        eigenvalues = eigenvalues[eigenvalues > self.min_eigenvalue]
        
        if len(eigenvalues) == 0:
            return 0
            
        # Apply variance fraction if specified
        if self.variance_fraction is not None:
            total_variance = eigenvalues.sum()
            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / total_variance
            n_comp = torch.searchsorted(cumulative_variance, self.variance_fraction).item() + 1
            return min(n_comp, len(eigenvalues))
        elif self.n_components is not None:
            return min(self.n_components, len(eigenvalues))
        else:
            return len(eigenvalues)


class DiagonalMahalanobisDistance(DistanceMetric):
    """Mahalanobis distance with diagonal covariance.
    
    More efficient special case where covariance is diagonal.
    Used in algorithms that assume feature independence within clusters.
    """
    
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute diagonal Mahalanobis distances.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean' and 'variance' (diagonal elements)
            
        Returns:
            (n,) tensor of distances
        """
        params = representation.get_parameters()
        
        if 'mean' not in params:
            raise ValueError("Distance requires 'mean' parameter")
        if 'variance' not in params:
            raise ValueError("Diagonal Mahalanobis requires 'variance' parameter")
            
        mean = params['mean']
        variance = params['variance']
        
        # Center points
        centered = points - mean.unsqueeze(0)
        
        # For diagonal case: (x - μ)^T Σ^(-1) (x - μ) = sum_i (x_i - μ_i)² / σ_i²
        # Avoid division by zero
        safe_variance = torch.clamp(variance, min=1e-8)
        
        if variance.dim() == 0:
            # Scalar variance (isotropic)
            distances = torch.sum(centered * centered, dim=1) / safe_variance
        else:
            # Vector variance (diagonal)
            distances = torch.sum(centered * centered / safe_variance.unsqueeze(0), dim=1)
            
        return distances


class LowRankMahalanobisDistance(DistanceMetric):
    """Mahalanobis distance for low-rank plus diagonal covariance.
    
    Efficient computation for covariance of form Σ = WW^T + σ²I
    using the Woodbury matrix identity. Used in PPCA-based methods.
    """
    
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute Mahalanobis distance with low-rank covariance.
        
        Uses Woodbury formula for efficient computation without
        forming the full covariance matrix.
        
        Args:
            points: (n, d) tensor of points
            representation: Must have 'mean', 'W' (factor loadings), and 'variance'
            
        Returns:
            (n,) tensor of distances
        """
        params = representation.get_parameters()
        
        required = ['mean', 'W', 'variance']
        for param in required:
            if param not in params:
                raise ValueError(f"Low-rank Mahalanobis requires '{param}' parameter")
                
        mean = params['mean']
        W = params['W']  # (d, r)
        variance = params['variance']  # scalar
        
        # Center points
        centered = points - mean.unsqueeze(0)  # (n, d)
        
        if W.shape[1] == 0:
            # No factors - just scaled Euclidean
            return torch.sum(centered * centered, dim=1) / variance
            
        # Use Woodbury identity:
        # (WW^T + σ²I)^(-1) = (1/σ²)I - (1/σ²)W(I + W^TW/σ²)^(-1)W^T(1/σ²)
        
        # First term: (1/σ²)||x - μ||²
        term1 = torch.sum(centered * centered, dim=1) / variance
        
        # For second term, compute W^T(x - μ) / σ
        Wt_centered = torch.matmul(centered, W) / torch.sqrt(variance)  # (n, r)
        
        # Compute M = I + W^TW/σ²
        WtW = torch.matmul(W.t(), W)
        M = torch.eye(W.shape[1], device=W.device) + WtW / variance
        
        # Solve M y = W^T(x - μ) / σ
        try:
            # Cholesky is fastest for positive definite
            L = torch.linalg.cholesky(M)
            y = torch.linalg.solve_triangular(L, Wt_centered.t(), upper=False)
            
            # Second term: y^T y / σ²
            term2 = torch.sum(y * y, dim=0) / variance
            
        except Exception:
            # Fallback to general solver
            M_inv = torch.linalg.pinv(M)
            y = torch.matmul(Wt_centered, M_inv)  # (n, r)
            term2 = torch.sum(Wt_centered * y, dim=1) / variance
            
        distances = term1 - term2
        
        # Should be non-negative
        distances = torch.clamp(distances, min=0.0)
        
        return distances