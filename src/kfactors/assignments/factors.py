import torch
from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation
from ..assignments.ranked import HyperplaneRanker

class IndependentFactorAssignment(AssignmentStrategy):
    """Assignment strategy based on variance consumption model.
    
    Computes membership weights where closer features get higher base membership,
    but similar features are penalized to avoid double-counting explained variance.
    """
    
    def __init__(self, base_claim: float = 1.0):
        """
        Args:
            base_claim: Base membership value for closest features
        """
        self.base_claim = base_claim
        self._ranker = HyperplaneRanker()
    
    @property
    def is_soft(self) -> bool:
        """Consumption membership produces soft-like assignments."""
        return True
        
    @property
    def is_ranked(self) -> bool:
        """This strategy uses ranking internally but returns membership."""
        return False
    
    def compute_rankings(self, points: Tensor, representation: ClusterRepresentation) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute distance-based rankings for a single representation.
        
        Args:
            points: (n, d) tensor of data points
            representation: Single EigenFactorRepresentation
            
        Returns:
            rankings: (n, K) tensor of feature indices ordered by distance
            aux_info: Dictionary with distances and other ranking info
        """
        return self._ranker.compute_assignments(points, [representation])

    def compute_memberships(self, points: Tensor, representation: ClusterRepresentation, 
                        rankings: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute membership weights from rankings using independence penalty.
        
        Args:
            points: (n, d) tensor of data points
            representation: Single EigenFactorRepresentation
            rankings: (n, K) tensor of feature indices ordered by distance
            
        Returns:
            membership: (n, K) tensor of membership weights per point
            aux_info: Dictionary with remaining variance and other info
        """
        n_points, n_factors = rankings.shape
        vectors = representation.vectors  # (K, d)
        
        # Initialize membership matrix and variance tracking
        membership = torch.zeros(n_points, n_factors, device=points.device)
        remaining_variance = torch.ones(n_points, device=points.device)
        
        # Process each ranking position
        for rank_pos in range(n_factors):
            for point_i in range(n_points):
                feature_j = rankings[point_i, rank_pos]
                
                # Available variance for this point
                available = remaining_variance[point_i]
                if available <= 1e-6:
                    continue
                
                # Base claim (fraction of remaining variance)
                if rank_pos == 0:
                    base_claim = min(self.base_claim, available)
                else:
                    base_claim = available * 0.5  # Could be tuned
                
                # Compute independence penalty from earlier features
                penalty = 1.0
                for earlier_rank in range(rank_pos):
                    earlier_feature = rankings[point_i, earlier_rank]
                    earlier_membership = membership[point_i, earlier_feature]
                    
                    if earlier_membership > 1e-6:
                        similarity = torch.abs(torch.cosine_similarity(
                            vectors[feature_j], 
                            vectors[earlier_feature], 
                            dim=0
                        )).item()
                        
                        penalty *= (1.0 - similarity * earlier_membership)
                
                # Final membership
                final_membership = base_claim * max(penalty, 0.0)
                membership[point_i, feature_j] = final_membership
                remaining_variance[point_i] -= final_membership
        
        aux_info = {
            'remaining_variance': remaining_variance.clone()
        }
        
        return membership, aux_info

    def compute_assignments(self, points: Tensor,
                        representations: List[ClusterRepresentation],
                        **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute membership weights using independence-based factor assignment.
        
        Args:
            points: (n, d) tensor of data points
            representations: Single EigenFactorRepresentation in a list
            **kwargs: Additional parameters
            
        Returns:
            membership: (n, K) tensor of membership weights per point
            aux_info: Dictionary with rankings, distances, etc.
        """
        if len(representations) != 1:
            raise ValueError("IndependentFactorAssignment expects exactly one representation")
        
        representation = representations[0]
        
        # Compute rankings
        rankings, rank_aux = self.compute_rankings(points, representation)
        
        # Compute memberships from rankings
        membership, membership_aux = self.compute_memberships(points, representation, rankings)
        
        # Combine auxiliary information
        aux_info = {
            'rankings': rankings,
            'distances': rank_aux['distances'],
            'raw_distances': rank_aux['raw_distances'],
            **membership_aux
        }
        
        return membership, aux_info
