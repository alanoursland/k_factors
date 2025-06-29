"""
Probabilistic PCA representation for K-Factors and C-Factors.

Represents clusters using a low-rank Gaussian model with:
- Mean μ
- Factor loading matrix W (d × r)
- Isotropic noise variance σ²

The covariance is implicitly: C = WW^T + σ²I
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from .base_representation import BaseRepresentation
from ..utils.linalg import orthogonalize_basis, safe_svd, safe_eigh


class PPCARepresentation(BaseRepresentation):
    """Probabilistic PCA cluster representation.
    
    Models each cluster as a Gaussian with low-rank covariance structure:
    x ~ N(μ, WW^T + σ²I)
    
    where W is a d×r factor loading matrix and σ² is isotropic noise.
    """
    
    def __init__(self, dimension: int, latent_dim: int, device: torch.device,
                 init_variance: float = 1.0):
        """
        Args:
            dimension: Ambient dimension d
            latent_dim: Latent dimension r (r < d)
            device: Torch device
            init_variance: Initial noise variance σ²
        """
        super().__init__(dimension, device)
        
        if latent_dim >= dimension:
            raise ValueError(f"Latent dimension {latent_dim} must be less than "
                           f"ambient dimension {dimension}")
                           
        self.latent_dim = latent_dim
        
        # Initialize factor loadings with random orthonormal columns
        if latent_dim > 0:
            W_init = torch.randn(dimension, latent_dim, device=device)
            self._W, _ = torch.linalg.qr(W_init)
            # Scale to have reasonable variance
            self._W = self._W * math.sqrt(init_variance)
        else:
            self._W = torch.empty(dimension, 0, device=device)
            
        # Initialize noise variance
        self._variance = torch.tensor(init_variance, device=device)
        
        # Cache for efficient computation
        self._M_inv = None  # Inverse of (I + W^T W / σ²)
        self._needs_cache_update = True
        
    @property
    def W(self) -> Tensor:
        """Factor loading matrix (d, r)."""
        return self._W
        
    @W.setter
    def W(self, value: Tensor):
        """Set factor loadings."""
        assert value.shape == (self._dimension, self.latent_dim)
        self._W = value.to(self._device)
        self._needs_cache_update = True
        
    @property
    def variance(self) -> Tensor:
        """Isotropic noise variance σ²."""
        return self._variance
        
    @variance.setter
    def variance(self, value: Tensor):
        """Set noise variance."""
        self._variance = torch.clamp(value.to(self._device), min=1e-6)
        self._needs_cache_update = True
        
    def _update_cache(self):
        """Update cached matrices for efficient computation."""
        if not self._needs_cache_update:
            return
            
        if self.latent_dim > 0:
            # Compute M = I + W^T W / σ²
            WtW = torch.matmul(self._W.t(), self._W)
            M = torch.eye(self.latent_dim, device=self._device) + WtW / self._variance
            
            # Compute M^(-1) for Woodbury formula
            try:
                self._M_inv = torch.linalg.inv(M)
            except:
                # Fallback to pseudo-inverse for numerical stability
                self._M_inv = torch.linalg.pinv(M)
        else:
            self._M_inv = None
            
        self._needs_cache_update = False
        
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """Compute Mahalanobis distance under the PPCA model.
        
        Uses the Woodbury formula for efficient computation:
        (WW^T + σ²I)^(-1) = (1/σ²)I - (1/σ²)W(I + W^TW/σ²)^(-1)W^T(1/σ²)
        
        Args:
            points: (n, d) data points
            indices: Ignored for standard PPCA
            
        Returns:
            (n,) squared Mahalanobis distances
        """
        self._check_points_shape(points)
        self._update_cache()
        
        # Center points
        centered = points - self._mean.unsqueeze(0)  # (n, d)
        
        if self.latent_dim == 0:
            # Pure isotropic Gaussian
            distances = torch.sum(centered * centered, dim=1) / self._variance
            return distances
            
        # Efficient Mahalanobis distance using Woodbury formula
        # First term: (1/σ²)||x - μ||²
        term1 = torch.sum(centered * centered, dim=1) / self._variance
        
        # Second term involves: W(I + W^TW/σ²)^(-1)W^T
        # Compute W^T(x - μ) / σ
        Wt_centered = torch.matmul(centered, self._W) / torch.sqrt(self._variance)  # (n, r)
        
        # Apply M^(-1)
        M_inv_Wt_centered = torch.matmul(Wt_centered, self._M_inv)  # (n, r)
        
        # Second term: subtract the correction
        term2 = torch.sum(Wt_centered * M_inv_Wt_centered, dim=1) / self._variance
        
        distances = term1 - term2
        
        return distances
        
    def log_likelihood(self, points: Tensor) -> Tensor:
        """Compute log-likelihood of points under this PPCA model.
        
        Args:
            points: (n, d) data points
            
        Returns:
            (n,) log-likelihood values
        """
        self._check_points_shape(points)
        
        # Get Mahalanobis distances
        distances = self.distance_to_point(points)
        
        # Log determinant of covariance using matrix determinant lemma
        if self.latent_dim > 0:
            # log|WW^T + σ²I| = d*log(σ²) + log|I + W^TW/σ²|
            WtW = torch.matmul(self._W.t(), self._W)
            M = torch.eye(self.latent_dim, device=self._device) + WtW / self._variance
            sign, logdet_M = torch.linalg.slogdet(M)
            log_det_cov = self._dimension * torch.log(self._variance) + logdet_M
        else:
            log_det_cov = self._dimension * torch.log(self._variance)
            
        # Gaussian log-likelihood
        log_likelihood = -0.5 * (distances + log_det_cov + self._dimension * math.log(2 * math.pi))
        
        return log_likelihood
        
    def sample_latent(self, n_samples: int) -> Tensor:
        """Sample from latent distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            (n_samples, r) latent samples from N(0, I)
        """
        return torch.randn(n_samples, self.latent_dim, device=self._device)
        
    def generate_samples(self, n_samples: int) -> Tensor:
        """Generate samples from this PPCA model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            (n_samples, d) generated samples
        """
        # Sample latent variables
        z = self.sample_latent(n_samples)
        
        # Generate x = μ + Wz + ε
        samples = self._mean.unsqueeze(0) + torch.matmul(z, self._W.t())
        
        # Add noise
        noise = torch.randn(n_samples, self._dimension, device=self._device)
        samples = samples + torch.sqrt(self._variance) * noise
        
        return samples
        
    def posterior_mean_cov(self, point: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute posterior distribution of latent variable given observation.
        
        For PPCA, the posterior is:
        p(z|x) = N(μ_z|x, Σ_z|x)
        where:
        Σ_z|x = (I + W^TW/σ²)^(-1) = M^(-1)
        μ_z|x = M^(-1)W^T(x - μ)/σ²
        
        Args:
            point: (d,) single observation
            
        Returns:
            posterior_mean: (r,) posterior mean
            posterior_cov: (r, r) posterior covariance
        """
        assert point.shape == (self._dimension,)
        self._update_cache()
        
        if self.latent_dim == 0:
            return torch.empty(0, device=self._device), torch.empty(0, 0, device=self._device)
            
        # Center point
        centered = point - self._mean
        
        # Posterior covariance is M^(-1)
        posterior_cov = self._M_inv
        
        # Posterior mean
        Wt_centered = torch.matmul(self._W.t(), centered) / self._variance
        posterior_mean = torch.matmul(self._M_inv, Wt_centered)
        
        return posterior_mean, posterior_cov
        
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None,
                          **kwargs) -> None:
        """Update PPCA parameters using closed-form ML or EM update.
        
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
            n_points = points.shape[0]
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
            
            # Apply weights for covariance computation
            centered = centered * torch.sqrt(normalized_weights).unsqueeze(1)
            n_points = total_weight
            
        if self.latent_dim == 0:
            # Just update variance
            self._variance = torch.sum(centered * centered) / (n_points * self._dimension)
            return
            
        # Compute sample covariance
        S = torch.matmul(centered.t(), centered) / n_points
        
        # Extract principal components
        try:
            eigvals, eigvecs = torch.linalg.eigh(S)
            
            # Sort in descending order
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Update W and variance using Tipping & Bishop formulas
            if self.latent_dim < self._dimension:
                # Top eigenvalues for the latent space
                lambda_r = eigvals[:self.latent_dim]
                U_r = eigvecs[:, :self.latent_dim]
                
                # Estimate noise variance from remaining eigenvalues
                if self.latent_dim < self._dimension - 1:
                    self._variance = torch.mean(eigvals[self.latent_dim:])
                else:
                    # Not enough eigenvalues - use smallest one
                    self._variance = eigvals[-1]
                    
                # Ensure variance is positive
                self._variance = torch.clamp(self._variance, min=1e-6)
                
                # Update W
                scale = torch.sqrt(torch.clamp(lambda_r - self._variance, min=1e-6))
                self._W = U_r * scale.unsqueeze(0)
                
            else:
                # Degenerate case: latent_dim >= d-1
                self._W = eigvecs[:, :self.latent_dim]
                self._variance = torch.tensor(1e-6, device=self._device)
                
        except Exception as e:
            import warnings
            warnings.warn(f"Eigendecomposition failed in PPCA update: {e}")
            
        self._needs_cache_update = True
        
    def get_parameters(self) -> Dict[str, Tensor]:
        """Get all parameters."""
        return {
            'mean': self._mean.clone(),
            'W': self._W.clone(),
            'variance': self._variance.clone()
        }
        
    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set parameters."""
        if 'mean' in params:
            self.mean = params['mean']
        if 'W' in params:
            self.W = params['W']
        if 'variance' in params:
            self.variance = params['variance']
            
    def to(self, device: torch.device) -> 'PPCARepresentation':
        """Move to device."""
        new_repr = PPCARepresentation(
            self._dimension, 
            self.latent_dim, 
            device,
            init_variance=self._variance.item()
        )
        
        params = self.get_parameters()
        new_params = {k: v.to(device) for k, v in params.items()}
        new_repr.set_parameters(new_params)
        
        return new_repr
        
    def __repr__(self) -> str:
        return (f"PPCARepresentation(dimension={self._dimension}, "
                f"latent_dim={self.latent_dim}, variance={self._variance.item():.3f})")