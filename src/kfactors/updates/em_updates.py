"""
EM update strategies for soft clustering algorithms.

Implements M-step updates for various probabilistic models including
Gaussian Mixture Models and Mixture of PPCA.
"""

from typing import Optional, Dict, Any
import torch
from torch import Tensor

from ..base.interfaces import ParameterUpdater, ClusterRepresentation
from ..representations.gaussian import GaussianRepresentation
from ..representations.ppca import PPCARepresentation
from ..utils.linalg import safe_eigh


class GaussianEMUpdater(ParameterUpdater):
    """EM updater for Gaussian representations.
    
    Implements the M-step for Gaussian mixture models with
    various covariance structures.
    """
    
    def __init__(self, min_covar: float = 1e-6,
                 covariance_type: str = 'full'):
        """
        Args:
            min_covar: Minimum variance for numerical stability
            covariance_type: Type of covariance ('full', 'diagonal', 'spherical', 'tied')
        """
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self._shared_covariance = None
        
"""
EM update strategies for soft clustering algorithms.

Implements M-step updates for various probabilistic models including
Gaussian Mixture Models and Mixture of PPCA.
"""

from typing import Optional, Dict, Any
import torch
from torch import Tensor

from ..base.interfaces import ParameterUpdater, ClusterRepresentation
from ..representations.gaussian import GaussianRepresentation
from ..representations.ppca import PPCARepresentation
from ..utils.linalg import safe_eigh


class GaussianEMUpdater(ParameterUpdater):
    """EM updater for Gaussian representations.
    
    Implements the M-step for Gaussian mixture models with
    various covariance structures.
    """
    
    def __init__(self, min_covar: float = 1e-6,
                 covariance_type: str = 'full'):
        """
        Args:
            min_covar: Minimum variance for numerical stability
            covariance_type: Type of covariance ('full', 'diagonal', 'spherical', 'tied')
        """
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self._shared_covariance = None
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignment_weights: Tensor,
               all_points: Optional[Tensor] = None,
               all_assignments: Optional[Tensor] = None,
               **kwargs) -> None:
        """Update Gaussian parameters using EM.
        
        Args:
            representation: GaussianRepresentation to update
            points: (n, d) points (subset for this cluster if hard assignment)
            assignment_weights: (n,) soft responsibilities for these points
            all_points: For tied covariance, all data points
            all_assignments: For tied covariance, all responsibilities (n, K)
            **kwargs: Additional info from E-step
        """
        if not isinstance(representation, GaussianRepresentation):
            raise TypeError("GaussianEMUpdater requires GaussianRepresentation")
            
        # For tied covariance, we need to compute shared covariance first
        if self.covariance_type == 'tied' and all_points is not None:
            self._update_shared_covariance(all_points, all_assignments, **kwargs)
            kwargs['shared_covariance'] = self._shared_covariance
            
        # Delegate to representation's update method
        representation.update_from_points(points, weights=assignment_weights, **kwargs)
        
    def _update_shared_covariance(self, points: Tensor, responsibilities: Tensor,
                                 representations: list = None, **kwargs):
        """Compute shared covariance for tied model.
        
        Args:
            points: (n, d) all data points
            responsibilities: (n, K) all responsibilities
            representations: List of all cluster representations
        """
        n_points, n_clusters = responsibilities.shape
        d = points.shape[1]
        device = points.device
        
        # Compute weighted covariance across all clusters
        total_covariance = torch.zeros(d, d, device=device)
        total_weight = 0.0
        
        if representations is not None:
            # Use current means
            for k, rep in enumerate(representations):
                weights = responsibilities[:, k]
                weight_sum = weights.sum()
                
                if weight_sum > 1e-10:
                    mean = rep.mean
                    centered = points - mean.unsqueeze(0)
                    weighted_centered = centered * torch.sqrt(weights).unsqueeze(1)
                    
                    cov_k = torch.matmul(weighted_centered.t(), weighted_centered)
                    total_covariance += cov_k
                    total_weight += weight_sum
        else:
            # Compute means first
            for k in range(n_clusters):
                weights = responsibilities[:, k]
                weight_sum = weights.sum()
                
                if weight_sum > 1e-10:
                    normalized_weights = weights / weight_sum
                    mean = torch.sum(points * normalized_weights.unsqueeze(1), dim=0)
                    centered = points - mean.unsqueeze(0)
                    weighted_centered = centered * torch.sqrt(weights).unsqueeze(1)
                    
                    cov_k = torch.matmul(weighted_centered.t(), weighted_centered)
                    total_covariance += cov_k
                    total_weight += weight_sum
                    
        # Normalize and regularize
        if total_weight > 0:
            self._shared_covariance = total_covariance / total_weight
            self._shared_covariance += self.min_covar * torch.eye(d, device=device)
        else:
            self._shared_covariance = torch.eye(d, device=device)


class PPCAEMUpdater(ParameterUpdater):
    """EM updater for Probabilistic PCA representations.
    
    Implements the M-step for Mixture of PPCA (C-Factors).
    """
    
    def __init__(self, n_components: int, min_variance: float = 1e-6):
        """
        Args:
            n_components: Number of latent dimensions
            min_variance: Minimum noise variance
        """
        self.n_components = n_components
        self.min_variance = min_variance
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignment_weights: Tensor,
               **kwargs) -> None:
        """Update PPCA parameters using EM.
        
        Uses the closed-form M-step updates from Tipping & Bishop.
        
        Args:
            representation: PPCARepresentation to update
            points: (n, d) points assigned to this cluster
            assignment_weights: (n,) soft responsibilities
        """
        if not isinstance(representation, PPCARepresentation):
            raise TypeError("PPCAEMUpdater requires PPCARepresentation")
            
        if len(points) == 0:
            return
            
        # Normalize responsibilities
        total_resp = assignment_weights.sum()
        if total_resp <= 1e-10:
            return
            
        normalized_resp = assignment_weights / total_resp
        
        # Update mean
        representation.mean = torch.sum(points * normalized_resp.unsqueeze(1), dim=0)
        
        # Center data
        centered = points - representation.mean.unsqueeze(0)
        
        # Compute weighted sample covariance
        weighted_centered = centered * torch.sqrt(normalized_resp).unsqueeze(1)
        S = torch.matmul(weighted_centered.t(), weighted_centered)
        
        # Eigendecomposition
        eigvals, eigvecs = safe_eigh(S)
        
        # Sort in descending order
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        d = points.shape[1]
        
        if self.n_components < d:
            # Standard case: extract top components
            U_r = eigvecs[:, :self.n_components]
            lambda_r = eigvals[:self.n_components]
            
            # Estimate noise variance from minor components
            if self.n_components < d - 1:
                sigma_sq = torch.mean(eigvals[self.n_components:])
            else:
                sigma_sq = eigvals[-1]
                
            sigma_sq = torch.clamp(sigma_sq, min=self.min_variance)
            
            # Update W using Tipping & Bishop formula
            # W = U_r * (Lambda_r - sigma^2 I)^{1/2}
            scale = torch.sqrt(torch.clamp(lambda_r - sigma_sq, min=1e-8))
            representation.W = U_r * scale.unsqueeze(0)
            representation.variance = sigma_sq
            
        else:
            # Degenerate case: n_components >= d
            representation.W = eigvecs[:, :self.n_components]
            representation.variance = torch.tensor(self.min_variance, device=points.device)


class MixingWeightUpdater(ParameterUpdater):
    """Updates mixing weights for mixture models.
    
    This is technically not a representation updater but fits
    the same interface for consistency.
    """
    
    def __init__(self, min_weight: float = 1e-10):
        """
        Args:
            min_weight: Minimum mixing weight
        """
        self.min_weight = min_weight
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignment_weights: Tensor,
               total_points: Optional[int] = None,
               **kwargs) -> None:
        """Update mixing weight based on responsibilities.
        
        The mixing weight π_k = N_k / N where N_k is the
        effective number of points in cluster k.
        
        Args:
            representation: Not used (for interface compatibility)
            points: Not used
            assignment_weights: (n,) responsibilities for this cluster
            total_points: Total number of data points
        """
        # Compute effective count
        n_k = assignment_weights.sum()
        
        # Total points defaults to sum of all assignments
        if total_points is None:
            total_points = len(assignment_weights)
            
        # Update mixing weight
        mixing_weight = torch.clamp(n_k / total_points, min=self.min_weight)
        
        # Store in kwargs for retrieval by algorithm
        if 'mixing_weights' not in kwargs:
            kwargs['mixing_weights'] = {}
        kwargs['mixing_weights'][id(representation)] = mixing_weight
        
        return mixing_weight