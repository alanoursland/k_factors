"""
Sequential PCA update strategy for K-Factors.

Extracts principal components sequentially from residuals after removing
previously extracted dimensions.
"""

from typing import Optional, Dict, Any
import torch
from torch import Tensor

from ..base.interfaces import ParameterUpdater, ClusterRepresentation
from ..representations.ppca import PPCARepresentation
from ..representations.subspace import SubspaceRepresentation
from ..utils.linalg import safe_svd, orthogonalize_basis


class SequentialPCAUpdater(ParameterUpdater):
    """Updates cluster representation by extracting next principal component.
    
    At each stage, performs PCA on the residuals after projecting out
    previously extracted dimensions, ensuring orthogonal basis vectors.
    """
    
    def __init__(self, current_stage: int = 0):
        """
        Args:
            current_stage: Which dimension is being extracted (0-indexed)
        """
        self.current_stage = current_stage
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignment_weights: Optional[Tensor] = None,
               current_stage: Optional[int] = None,
               **kwargs) -> None:
        """Update representation with next principal component.
        
        Args:
            representation: PPCA or Subspace representation to update
            points: Points assigned to this cluster
            assignment_weights: Weighted contribution of points to each feature
            current_stage: Override stage if provided
            **kwargs: Additional info from assignment
        """
        if current_stage is not None:
            stage = current_stage
        else:
            stage = self.current_stage
            
        if len(points) == 0:
            return
            
        if assignment_weights is not None:
            w = assignment_weights.to(points.device).clamp_min(0.0)
            w_sum = w.sum()
            if w_sum.item() > 0:
                w_norm = w / w_sum
                # Weighted mean
                representation.mean = torch.sum(points * w_norm.unsqueeze(1), dim=0)
                centered = points - representation.mean.unsqueeze(0)
                # Apply sqrt-weights for covariance/SVD steps
                centered = centered * torch.sqrt(w_norm).unsqueeze(1)
            else:
                # No effective weight: keep previous mean; nothing to do
                return
        else:
            # Unweighted path
            representation.mean = points.mean(dim=0)
            centered = points - representation.mean.unsqueeze(0)

        # Update mean (always use latest assigned points)
        representation.mean = points.mean(dim=0)
        centered = points - representation.mean.unsqueeze(0)
        
        if isinstance(representation, PPCARepresentation):
            self._update_ppca(representation, centered, stage, assignment_weights)
        elif isinstance(representation, SubspaceRepresentation):
            self._update_subspace(representation, centered, stage)
        else:
            raise TypeError(f"Sequential PCA updater requires PPCA or Subspace representation, "
                          f"got {type(representation)}")
                          
    def _update_ppca(self, 
                     representation: PPCARepresentation, 
                     centered: Tensor, 
                     stage: int,
                     assignments: Optional[Tensor] = None) -> None:
        """Update PPCA representation sequentially."""
        d = centered.shape[1]
        
        # Initialize W if needed
        if stage == 0:
            # First stage - initialize W matrix
            representation._W = torch.zeros(d, representation.latent_dim, 
                                          device=centered.device)
                                          
        # Compute residuals after removing previous components
        if stage > 0:
            # Project out previous directions
            prev_basis = representation.W[:, :stage]
            coeffs = torch.matmul(centered, prev_basis)
            projections = torch.matmul(coeffs, prev_basis.t())
            residuals = centered - projections
        else:
            residuals = centered
            
        # Compute covariance of residuals
        # n_eff: total weight if weighted, else n_samples
        if assignments is not None:
            w = assignments.to(residuals.device).clamp_min(0.0)
            n = w.sum()
            cov = torch.matmul(residuals.t(), residuals) / (n + 1e-12)
        else:
            n = residuals.shape[0]
            cov = torch.matmul(residuals.t(), residuals) / n

        # Extract top eigenvector
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # Take largest eigenvalue/vector
            idx = torch.argmax(eigvals)
            top_eigval = eigvals[idx]
            top_eigvec = eigvecs[:, idx]
            
            # Ensure consistent sign (for reproducibility)
            if top_eigvec[0] < 0:
                top_eigvec = -top_eigvec
                
        except Exception:
            # Fallback to SVD
            U, S, Vt = safe_svd(residuals, full_matrices=False)
            top_eigvec = Vt[0]
            top_eigval = (S[0] ** 2) / n
            
        # Update W at current stage
        if stage < representation.latent_dim:
            # Scale by sqrt(eigenvalue - noise variance)
            # For sequential extraction, we update noise variance later
            scale = torch.sqrt(torch.clamp(top_eigval, min=1e-6))
            representation._W[:, stage] = top_eigvec * scale
            
            # Orthogonalize against previous columns (numerical safety)
            if stage > 0:
                representation._W[:, :stage+1] = orthogonalize_basis(
                    representation._W[:, :stage+1]
                )
                
        # Update noise variance (average residual variance)
        if stage == representation.latent_dim - 1:
            # Final stage - compute noise from final residuals
            final_basis = representation.W
            final_coeffs = torch.matmul(centered, final_basis)
            final_projections = torch.matmul(final_coeffs, final_basis.t())
            final_residuals = centered - final_projections
            
            representation.variance = torch.mean(
                torch.sum(final_residuals * final_residuals, dim=1)
            ) / d
            
        representation._needs_cache_update = True
        
    def _update_subspace(self, representation: SubspaceRepresentation,
                        centered: Tensor, stage: int) -> None:
        """Update subspace representation sequentially."""
        d = centered.shape[1]
        
        # Initialize basis if needed
        if stage == 0:
            representation._basis = torch.zeros(d, representation.subspace_dim,
                                              device=centered.device)
                                              
        # Compute residuals
        if stage > 0:
            prev_basis = representation.basis[:, :stage]
            coeffs = torch.matmul(centered, prev_basis)
            projections = torch.matmul(coeffs, prev_basis.t())
            residuals = centered - projections
        else:
            residuals = centered
            
        # Extract top principal component of residuals
        try:
            # Use SVD for better numerical stability
            U, S, Vt = safe_svd(residuals, full_matrices=False)
            top_direction = Vt[0]
            
            # Ensure consistent sign
            if top_direction[0] < 0:
                top_direction = -top_direction
                
        except Exception:
            # Fallback to covariance method
            n = residuals.shape[0]
            cov = torch.matmul(residuals.t(), residuals) / n
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = torch.argmax(eigvals)
            top_direction = eigvecs[:, idx]
            
        # Update basis at current stage
        if stage < representation.subspace_dim:
            representation._basis[:, stage] = top_direction
            
            # Orthogonalize for numerical stability
            if stage > 0:
                representation._basis[:, :stage+1] = orthogonalize_basis(
                    representation._basis[:, :stage+1]
                )