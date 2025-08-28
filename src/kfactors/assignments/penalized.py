"""
Penalized assignment strategy for K-Factors algorithm.

Assigns points to clusters with a penalty for reusing previously claimed directions.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import Tensor

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation
from ..base.data_structures import DirectionTracker
from ..representations.ppca import PPCARepresentation
from ..representations.eigenfactor import EigenFactorRepresentation
from ..representations.subspace import SubspaceRepresentation


class PenalizedAssignment(AssignmentStrategy):
    """Assignment strategy with penalty for claimed directions.
    
    Used in K-Factors to ensure each point claims diverse directions
    across iterations, preventing collapse to a single subspace.
    """
    
    def __init__(self, penalty_type: str = 'product', 
                 penalty_weight: float = 1.0):
        """
        Args:
            penalty_type: 'product' or 'sum' penalty computation
            penalty_weight: Weight for penalty term (higher = stronger penalty)
        """
        super().__init__()
        self.penalty_type = penalty_type
        self.penalty_weight = penalty_weight
        
    @property
    def is_soft(self) -> bool:
        """Penalized assignments are hard."""
        return False
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          direction_tracker: Optional[DirectionTracker] = None,
                          current_stage: int = 0,
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute penalized assignments.
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            direction_tracker: Tracks claimed directions per point
            current_stage: Current iteration/stage in K-Factors
            
        Returns:
            assignments: (n,) hard assignments
            aux_info: Dictionary with penalty information
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        device = points.device
        
        # Compute base distances (residual costs)
        distances = torch.zeros(n_points, n_clusters, device=device)
        current_directions = torch.zeros(n_points, n_clusters, points.shape[1], device=device)
        
        for k, representation in enumerate(representations):
            if isinstance(representation, PPCARepresentation):
                # For PPCA, use residual after projecting onto current basis
                # This is the distance to the subspace at current stage
                centered = points - representation.mean.unsqueeze(0)
                
                # Build an orthonormal basis for span(W) (do NOT mutate rep)
                W_full = representation.W
                if W_full.numel() > 0:
                    # QR gives Q with orthonormal columns spanning W
                    # (reduced mode by default)
                    Q, _ = torch.linalg.qr(W_full)
                else:
                    Q = W_full  # empty

                # Current stage uses the first (current_stage+1) orthonormal cols
                if current_stage < Q.shape[1]:
                    current_basis = Q[:, :current_stage+1]
                    
                    # Project onto current basis
                    coeffs = torch.matmul(centered, current_basis)
                    projections = torch.matmul(coeffs, current_basis.t())
                    
                    # Residual is what's left
                    residuals = centered - projections
                    distances[:, k] = torch.sum(residuals * residuals, dim=1)
                    
                    # Current direction used for penalty = unit column of Q
                    current_directions[:, k] = current_basis[:, -1]
                else:
                    # All dimensions used - just distance to full subspace
                    distances[:, k] = representation.distance_to_point(points)
                    
            elif isinstance(representation, SubspaceRepresentation):
                # For subspace representation with sequential extraction
                centered = points - representation.mean.unsqueeze(0)
                
                if current_stage < representation.basis.shape[1]:
                    # Use partial basis up to current stage
                    current_basis = representation.basis[:, :current_stage+1]
                    coeffs = torch.matmul(centered, current_basis)
                    projections = torch.matmul(coeffs, current_basis.t())
                    residuals = centered - projections
                    distances[:, k] = torch.sum(residuals * residuals, dim=1)
                    
                    # Current direction
                    current_directions[:, k] = representation.basis[:, current_stage]
                else:
                    distances[:, k] = representation.distance_to_point(points)

            elif isinstance(representation, EigenFactorRepresentation):
                # Flat factors: compute distances over all factors inside this single representation
                ef = representation
                Kf = ef.n_factors
                eps = 1e-12

                # (n, Kf): squared normalized residual to each hyperplane
                x_exp = points.unsqueeze(1)             # (n, 1, D)
                mu = ef.means.unsqueeze(0)              # (1, Kf, D)
                w  = ef.vectors.unsqueeze(0)            # (1, Kf, D)

                centered = x_exp - mu                   # (n, Kf, D)
                num = (centered * w).sum(dim=2)         # (n, Kf)
                den = (ef.vectors * ef.vectors).sum(dim=1).clamp_min(eps).unsqueeze(0)  # (1, Kf)
                distances_f = (num * num) / den         # (n, Kf)

                # directions for penalty (same vector per point/factor)
                current_dirs = ef.vectors.unsqueeze(0).expand(n_points, Kf, -1)  # (n, Kf, D)

                # --- minimal change: overwrite the preallocated tensors and cluster count ---
                distances = distances_f
                current_directions = current_dirs
                n_clusters = Kf  # ensure downstream penalty/argmin operate over factors
                break            # we handled the flat case; no other reps expected

            else:
                # Fallback for other representations
                distances[:, k] = representation.distance_to_point(points)
                
        # Apply penalties if direction tracker is provided
        if direction_tracker is not None:
            penalties = direction_tracker.compute_penalty_batch(
                current_directions, 
                penalty_type=self.penalty_type
            )
        else:
            penalties = torch.ones(n_points, n_clusters, device=device)
            
        # Assign to minimum penalized distance
        assignments = torch.argmin(distances, dim=1)
        
        # Prepare auxiliary information
        aux_info = {
            'distances': distances,
            'penalties': penalties,
            'current_directions': current_directions,
            'current_stage': current_stage,
            'min_distances': torch.gather(distances, 1, assignments.unsqueeze(1)).squeeze(1),
        }
        
        return assignments, aux_info