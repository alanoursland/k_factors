"""
Gaussian representation for clustering.

Represents clusters as full multivariate Gaussians with arbitrary covariance.
Used in Gaussian Mixture Models (GMM) and related algorithms.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from .base_representation import BaseRepresentation
from ..utils.linalg import safe_eigh, matrix_sqrt, solve_linear_system


class GaussianRepresentation(BaseRepresentation):
    """Full Gaussian cluster representation.
    
    Models each cluster as a multivariate Gaussian N(μ, Σ) with
    arbitrary covariance matrix.
    """
    
    def __init__(self, dimension: int, device: torch.device,
                 covariance_type: str = 'full',
                 regularization: float = 1e-6):
        """
        Args:
            dimension: Ambient dimension d
            device: Torch device
            covariance_type: Type of covariance matrix:
                - 'full': Arbitrary positive definite matrix
                - 'diagonal': Diagonal covariance
                - 'spherical': Scalar * identity
                - 'tied': Same covariance for all clusters (set externally)
            regularization: Small value added to diagonal for stability
        """
        super().__init__(dimension, device)
        
        self.covariance_type = covariance_type
        self.regularization = regularization
        
        # Initialize covariance based on type
        if covariance_type == 'full':
            self._covariance = torch.eye(dimension, device=device)
            self._precision = torch.eye(dimension, device=device)
            self._cholesky = torch.eye(dimension, device=device)
        elif covariance_type == 'diagonal':
            self._variance = torch.ones(dimension, device=device)
        elif covariance_type == 'spherical':
            self._variance = torch.tensor(1.0, device=device)
        elif covariance_type == 'tied':
            # Covariance will be set externally
            self._covariance = None
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")
            
        # Cache for efficient computation
        self._log_det = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._needs_update = True
        
    @property
    def covariance(self) -> Tensor:
        """Get covariance matrix."""
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            return self._covariance
        elif self.covariance_type == 'diagonal':
            return torch.diag(self._variance)
        elif self.covariance_type == 'spherical':
            return self._variance * torch.eye(self._dimension, device=self._device)
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")
            
    @covariance.setter
    def covariance(self, value: Tensor):
        """Set covariance matrix."""
        if self.covariance_type == 'full':
            assert value.shape == (self._dimension, self._dimension)
            self._covariance = value.to(self._device)
            self._needs_update = True
        elif self.covariance_type == 'tied':
            # External covariance for tied model
            self._covariance = value.to(self._device)
            self._needs_update = True
        else:
            raise ValueError(f"Cannot set full covariance for {self.covariance_type} type")
            
    @property
    def variance(self) -> Tensor:
        """Get variance (for diagonal/spherical types)."""
        if self.covariance_type == 'diagonal':
            return self._variance
        elif self.covariance_type == 'spherical':
            return self._variance
        elif self.covariance_type == 'full':
            return torch.diag(self._covariance)
        else:
            raise ValueError(f"Variance not defined for {self.covariance_type}")
            
    @variance.setter  
    def variance(self, value: Tensor):
        """Set variance."""
        if self.covariance_type == 'diagonal':
            assert value.shape == (self._dimension,)
            self._variance = torch.clamp(value.to(self._device), min=self.regularization)
        elif self.covariance_type == 'spherical':
            assert value.dim() == 0 or value.numel() == 1
            self._variance = torch.clamp(value.to(self._device).squeeze(), min=self.regularization)
        else:
            raise ValueError(f"Cannot set variance for {self.covariance_type} type")
            
    def _update_cache(self):
        """Update cached values for efficient computation."""
        if not self._needs_update:
            return
            
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            # Add regularization
            cov_reg = self._covariance + self.regularization * torch.eye(
                self._dimension, device=self._device
            )
            
            try:
                # Cholesky decomposition
                self._cholesky = torch.linalg.cholesky(cov_reg)
                
                # Log determinant via Cholesky
                self._log_det = 2 * torch.sum(torch.log(torch.diag(self._cholesky)))
                
                # Precision matrix (inverse covariance)
                eye = torch.eye(self._dimension, device=self._device)
                self._precision = torch.cholesky_solve(eye, self._cholesky)
                
            except Exception:
                # Fallback to eigendecomposition
                eigvals, eigvecs = safe_eigh(cov_reg)
                eigvals = torch.clamp(eigvals, min=self.regularization)
                
                self._eigenvalues = eigvals
                self._eigenvectors = eigvecs
                self._log_det = torch.sum(torch.log(eigvals))
                self._precision = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.t()
                
        elif self.covariance_type == 'diagonal':
            self._log_det = torch.sum(torch.log(self._variance))
            
        elif self.covariance_type == 'spherical':
            self._log_det = self._dimension * torch.log(self._variance)
            
        self._needs_update = False
        
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """Compute squared Mahalanobis distance.
        
        Args:
            points: (n, d) data points
            indices: Ignored
            
        Returns:
            (n,) squared Mahalanobis distances
        """
        self._check_points_shape(points)
        self._update_cache()
        
        centered = points - self._mean.unsqueeze(0)
        
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            # Use precision matrix if available
            if hasattr(self, '_precision') and self._precision is not None:
                # d = (x-μ)ᵀ Σ⁻¹ (x-μ)
                temp = torch.matmul(centered, self._precision.t())
                distances = torch.sum(centered * temp, dim=1)
            else:
                # Use Cholesky solve
                y = torch.linalg.solve_triangular(
                    self._cholesky, centered.t(), upper=False
                )
                distances = torch.sum(y * y, dim=0)
                
        elif self.covariance_type == 'diagonal':
            distances = torch.sum(centered ** 2 / self._variance.unsqueeze(0), dim=1)
            
        elif self.covariance_type == 'spherical':
            distances = torch.sum(centered ** 2, dim=1) / self._variance
            
        return distances
        
    def log_likelihood(self, points: Tensor) -> Tensor:
        """Compute log-likelihood of points under this Gaussian.
        
        Args:
            points: (n, d) data points
            
        Returns:
            (n,) log-likelihood values
        """
        self._check_points_shape(points)
        self._update_cache()
        
        # Mahalanobis distances
        distances = self.distance_to_point(points)
        
        # Gaussian log-likelihood
        # log p(x) = -0.5 * (d*log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ))
        constant = self._dimension * math.log(2 * math.pi)
        log_probs = -0.5 * (constant + self._log_det + distances)
        
        return log_probs
        
    def sample(self, n_samples: int) -> Tensor:
        """Generate samples from this Gaussian.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            (n_samples, d) samples
        """
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            # Sample z ~ N(0, I)
            z = torch.randn(n_samples, self._dimension, device=self._device)
            
            # Transform: x = μ + L z where Σ = L Lᵀ
            if hasattr(self, '_cholesky') and self._cholesky is not None:
                samples = self._mean + torch.matmul(z, self._cholesky.t())
            else:
                # Use eigendecomposition
                self._update_cache()
                L = self._eigenvectors @ torch.diag(torch.sqrt(self._eigenvalues))
                samples = self._mean + torch.matmul(z, L.t())
                
        elif self.covariance_type == 'diagonal':
            z = torch.randn(n_samples, self._dimension, device=self._device)
            samples = self._mean + z * torch.sqrt(self._variance)
            
        elif self.covariance_type == 'spherical':
            z = torch.randn(n_samples, self._dimension, device=self._device)
            samples = self._mean + z * torch.sqrt(self._variance)
            
        return samples
        
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None,
                          **kwargs) -> None:
        """Update Gaussian parameters from assigned points.
        
        Args:
            points: (n, d) assigned points
            weights: Optional (n,) weights for soft assignments
            **kwargs: May contain 'shared_covariance' for tied model
        """
        self._check_points_shape(points)
        
        if len(points) == 0:
            return
            
        # Update mean
        if weights is None:
            self._mean = points.mean(dim=0)
            centered = points - self._mean.unsqueeze(0)
            n_points = len(points)
        else:
            assert weights.shape == (points.shape[0],)
            weights = weights.to(self._device)
            total_weight = weights.sum()
            
            if total_weight <= 0:
                return
                
            normalized_weights = weights / total_weight
            self._mean = torch.sum(points * normalized_weights.unsqueeze(1), dim=0)
            centered = points - self._mean.unsqueeze(0)
            n_points = total_weight
            
        # Update covariance based on type
        if self.covariance_type == 'full':
            if weights is None:
                # Unweighted covariance
                cov = torch.matmul(centered.t(), centered) / n_points
            else:
                # Weighted covariance
                weighted_centered = centered * torch.sqrt(normalized_weights).unsqueeze(1)
                cov = torch.matmul(weighted_centered.t(), weighted_centered)
                
            # Add regularization
            self._covariance = cov + self.regularization * torch.eye(
                self._dimension, device=self._device
            )
            
        elif self.covariance_type == 'diagonal':
            if weights is None:
                var = torch.mean(centered ** 2, dim=0)
            else:
                var = torch.sum(centered ** 2 * normalized_weights.unsqueeze(1), dim=0)
            self._variance = torch.clamp(var, min=self.regularization)
            
        elif self.covariance_type == 'spherical':
            if weights is None:
                var = torch.mean(centered ** 2)
            else:
                var = torch.sum(centered ** 2 * normalized_weights.unsqueeze(1))
            self._variance = torch.clamp(var, min=self.regularization)
            
        elif self.covariance_type == 'tied':
            # Use shared covariance if provided
            if 'shared_covariance' in kwargs:
                self._covariance = kwargs['shared_covariance']
                
        self._needs_update = True
        
    def get_parameters(self) -> Dict[str, Tensor]:
        """Get all parameters."""
        params = {'mean': self._mean.clone()}
        
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            params['covariance'] = self.covariance.clone()
            if hasattr(self, '_precision') and self._precision is not None:
                params['precision'] = self._precision.clone()
            if hasattr(self, '_eigenvalues') and self._eigenvalues is not None:
                params['eigenvalues'] = self._eigenvalues.clone()
                params['eigenvectors'] = self._eigenvectors.clone()
        elif self.covariance_type == 'diagonal':
            params['variance'] = self._variance.clone()
        elif self.covariance_type == 'spherical':
            params['variance'] = self._variance.clone()
            
        return params
        
    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set parameters."""
        if 'mean' in params:
            self.mean = params['mean']
            
        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            if 'covariance' in params:
                self.covariance = params['covariance']
        elif self.covariance_type == 'diagonal' or self.covariance_type == 'spherical':
            if 'variance' in params:
                self.variance = params['variance']
                
    def get_ellipse_parameters(self, n_std: float = 2.0) -> Tuple[Tensor, Tensor, Tensor]:
        """Get parameters for plotting confidence ellipse.
        
        Args:
            n_std: Number of standard deviations
            
        Returns:
            center: (d,) center of ellipse
            radii: (d,) semi-axes lengths  
            rotation: (d, d) rotation matrix
        """
        self._update_cache()
        
        if self.covariance_type == 'full':
            if self._eigenvalues is None:
                eigvals, eigvecs = safe_eigh(self._covariance)
            else:
                eigvals, eigvecs = self._eigenvalues, self._eigenvectors
                
            radii = n_std * torch.sqrt(eigvals)
            rotation = eigvecs
            
        elif self.covariance_type == 'diagonal':
            radii = n_std * torch.sqrt(self._variance)
            rotation = torch.eye(self._dimension, device=self._device)
            
        elif self.covariance_type == 'spherical':
            radii = n_std * torch.sqrt(self._variance) * torch.ones(self._dimension, device=self._device)
            rotation = torch.eye(self._dimension, device=self._device)
            
        else:
            raise ValueError(f"Cannot compute ellipse for {self.covariance_type}")
            
        return self._mean, radii, rotation
        
    def __repr__(self) -> str:
        return (f"GaussianRepresentation(dimension={self._dimension}, "
                f"type={self.covariance_type})")