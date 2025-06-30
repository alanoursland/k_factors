"""
Soft (probabilistic) assignment strategies for clustering.

Used in algorithms like GMM, C-Factors, and EM-based methods where
points have fractional membership in multiple clusters.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import Tensor
import math

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation
from ..representations.gaussian import GaussianRepresentation
from ..representations.ppca import PPCARepresentation


class SoftAssignment(AssignmentStrategy):
    """Soft (probabilistic) assignment based on likelihood.
    
    Computes posterior probabilities of cluster membership using
    Bayes' rule with the cluster likelihoods and mixing weights.
    """
    
    def __init__(self, min_responsibility: float = 1e-10):
        """
        Args:
            min_responsibility: Minimum responsibility to avoid numerical issues
        """
        super().__init__()
        self.min_responsibility = min_responsibility
        
    @property
    def is_soft(self) -> bool:
        """Soft assignments are soft."""
        return True
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          mixing_weights: Optional[Tensor] = None,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute soft assignments (responsibilities).
        
        Uses Bayes' rule: p(k|x) = p(x|k)π_k / Σ_j p(x|j)π_j
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            mixing_weights: (K,) prior probabilities for each cluster
            
        Returns:
            responsibilities: (n, K) posterior probabilities
            aux_info: Dictionary with log-likelihood info
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        device = points.device
        
        # Default to uniform mixing weights
        if mixing_weights is None:
            mixing_weights = torch.ones(n_clusters, device=device) / n_clusters
        else:
            mixing_weights = mixing_weights.to(device)
            
        # Compute log-likelihoods for numerical stability
        log_likelihoods = torch.zeros(n_points, n_clusters, device=device)
        
        for k, representation in enumerate(representations):
            if hasattr(representation, 'log_likelihood'):
                # Use built-in log-likelihood method
                log_likelihoods[:, k] = representation.log_likelihood(points)
            else:
                # Fallback: assume Gaussian with unit variance
                distances = representation.distance_to_point(points)
                d = points.shape[1]
                log_likelihoods[:, k] = -0.5 * (distances + d * math.log(2 * math.pi))
                
        # Add log mixing weights
        log_weighted = log_likelihoods + torch.log(mixing_weights).unsqueeze(0)
        
        # Compute log marginal likelihood using log-sum-exp trick
        log_marginal = torch.logsumexp(log_weighted, dim=1, keepdim=True)
        
        # Compute responsibilities (posterior probabilities)
        log_responsibilities = log_weighted - log_marginal
        responsibilities = torch.exp(log_responsibilities)
        
        # Clamp to avoid numerical issues
        responsibilities = torch.clamp(responsibilities, min=self.min_responsibility)
        
        # Normalize to ensure they sum to 1
        responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)
        
        # Compute auxiliary information
        aux_info = {
            'log_likelihoods': log_likelihoods,
            'log_marginal': log_marginal.squeeze(1),
            'total_log_likelihood': log_marginal.sum().item(),
            'mixing_weights': mixing_weights,
            'effective_counts': responsibilities.sum(dim=0)
        }
        
        return responsibilities, aux_info


class GaussianSoftAssignment(SoftAssignment):
    """Soft assignment specifically for Gaussian representations.
    
    Optimized for Gaussian mixture models with various covariance types.
    """
    
    def compute_assignments(self, points: Tensor,
                          representations: List[GaussianRepresentation],
                          mixing_weights: Optional[Tensor] = None,
                          shared_covariance: Optional[Tensor] = None,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute Gaussian soft assignments.
        
        Args:
            points: (n, d) data points  
            representations: List of GaussianRepresentations
            mixing_weights: (K,) mixing weights
            shared_covariance: For tied covariance models
            
        Returns:
            responsibilities: (n, K) posterior probabilities
            aux_info: Dictionary with diagnostics
        """
        # Update tied covariances if needed
        if shared_covariance is not None:
            for rep in representations:
                if rep.covariance_type == 'tied':
                    rep.covariance = shared_covariance
                    
        # Use parent class implementation
        return super().compute_assignments(points, representations, mixing_weights, **kwargs)


class AnnealedSoftAssignment(SoftAssignment):
    """Soft assignment with temperature annealing.
    
    Uses a temperature parameter to control the "softness" of assignments.
    High temperature → uniform assignments
    Low temperature → nearly hard assignments
    """
    
    def __init__(self, temperature: float = 1.0, 
                 min_temperature: float = 0.1,
                 annealing_rate: float = 0.95,
                 min_responsibility: float = 1e-10):
        """
        Args:
            temperature: Initial temperature
            min_temperature: Minimum temperature
            annealing_rate: Temperature decay rate per iteration
            min_responsibility: Minimum responsibility value
        """
        super().__init__(min_responsibility)
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate
        self.initial_temperature = temperature
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          mixing_weights: Optional[Tensor] = None,
                          iteration: Optional[int] = None,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute annealed soft assignments.
        
        Args:
            points: (n, d) data points
            representations: Cluster representations
            mixing_weights: Prior probabilities
            iteration: Current iteration (for annealing schedule)
            
        Returns:
            responsibilities: (n, K) annealed posterior probabilities
            aux_info: Dictionary with temperature info
        """
        # Update temperature based on iteration
        if iteration is not None:
            self.temperature = max(
                self.min_temperature,
                self.initial_temperature * (self.annealing_rate ** iteration)
            )
            
        # Get base soft assignments
        responsibilities, aux_info = super().compute_assignments(
            points, representations, mixing_weights, **kwargs
        )
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            # Raise to power 1/T and renormalize
            responsibilities = torch.pow(responsibilities, 1.0 / self.temperature)
            responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)
            
        # Add temperature to aux info
        aux_info['temperature'] = self.temperature
        
        return responsibilities, aux_info
        
    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.temperature = self.initial_temperature


class EntropyRegularizedAssignment(SoftAssignment):
    """Soft assignment with entropy regularization.
    
    Adds an entropy term to encourage more uniform assignments,
    useful for preventing cluster collapse.
    """
    
    def __init__(self, entropy_weight: float = 0.1,
                 min_responsibility: float = 1e-10):
        """
        Args:
            entropy_weight: Weight for entropy regularization
            min_responsibility: Minimum responsibility
        """
        super().__init__(min_responsibility)
        self.entropy_weight = entropy_weight
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          mixing_weights: Optional[Tensor] = None,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute entropy-regularized assignments.
        
        Solves: min_q KL(q || p) - λH(q)
        where p is the base posterior and H(q) is entropy.
        
        Returns:
            responsibilities: (n, K) regularized posteriors
            aux_info: Dictionary with entropy values
        """
        # Get base assignments
        base_resp, aux_info = super().compute_assignments(
            points, representations, mixing_weights, **kwargs
        )
        
        if self.entropy_weight > 0:
            # Apply entropy regularization
            # Solution: q_k ∝ p_k^(1/(1+λ))
            scale = 1.0 / (1.0 + self.entropy_weight)
            regularized = torch.pow(base_resp, scale)
            regularized = regularized / regularized.sum(dim=1, keepdim=True)
            
            # Compute entropy for diagnostics
            entropy = -torch.sum(regularized * torch.log(regularized + 1e-10), dim=1)
            aux_info['entropy'] = entropy.mean().item()
            aux_info['base_responsibilities'] = base_resp
            
            return regularized, aux_info
        else:
            return base_resp, aux_info