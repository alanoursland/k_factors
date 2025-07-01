"""
Fuzzy assignment strategy for Fuzzy C-Means clustering.

Implements the fuzzy membership calculation where each point has
fractional membership in all clusters.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import Tensor

from ..base.interfaces import AssignmentStrategy, ClusterRepresentation


class FuzzyAssignment(AssignmentStrategy):
    """Fuzzy assignment for Fuzzy C-Means algorithm.
    
    Computes fuzzy memberships based on distances to cluster centers
    using the standard FCM membership formula.
    """
    
    def __init__(self, m: float = 2.0, epsilon: float = 1e-10):
        """
        Args:
            m: Fuzziness exponent (m > 1). Higher values make clustering fuzzier.
               m=1 gives hard clustering, m→∞ gives uniform memberships.
            epsilon: Small value to prevent division by zero
        """
        super().__init__()
        if m <= 1:
            raise ValueError(f"Fuzziness exponent m must be > 1, got {m}")
        self.m = m
        self.epsilon = epsilon
        
    @property
    def is_soft(self) -> bool:
        """Fuzzy assignments are soft."""
        return True
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute fuzzy memberships.
        
        Uses the formula:
        u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
        
        where d_ik is the distance from point i to cluster k.
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            
        Returns:
            memberships: (n, K) fuzzy membership matrix
            aux_info: Dictionary with distance information
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        device = points.device
        
        # Compute distances to all clusters
        distances = torch.zeros(n_points, n_clusters, device=device)
        
        for k, representation in enumerate(representations):
            distances[:, k] = representation.distance_to_point(points)
            
        # Add epsilon to prevent division by zero
        distances = distances + self.epsilon
        
        # Handle the case where a point is exactly at a cluster center
        # In this case, assign membership 1 to that cluster, 0 to others
        exact_matches = distances < self.epsilon * 10
        
        if exact_matches.any():
            # Create membership matrix
            memberships = torch.zeros_like(distances)
            
            # For exact matches, set membership to 1
            for i in range(n_points):
                exact_clusters = torch.where(exact_matches[i])[0]
                if len(exact_clusters) > 0:
                    # If multiple exact matches, distribute equally
                    memberships[i, exact_clusters] = 1.0 / len(exact_clusters)
                else:
                    # Use normal fuzzy calculation
                    memberships[i] = self._compute_fuzzy_row(distances[i])
        else:
            # Normal case: compute fuzzy memberships
            # Use the formula: u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
            power = 2.0 / (self.m - 1.0)
            
            # Compute membership matrix efficiently
            memberships = torch.zeros_like(distances)
            
            for i in range(n_points):
                memberships[i] = self._compute_fuzzy_row(distances[i])
                
        # Ensure memberships sum to 1 (handle numerical errors)
        row_sums = memberships.sum(dim=1, keepdim=True)
        memberships = memberships / row_sums.clamp(min=self.epsilon)
        
        # Compute auxiliary information
        aux_info = {
            'distances': distances,
            'exact_matches': exact_matches,
            'min_distances': distances.min(dim=1)[0],
            'fuzziness': self.m,
            'membership_entropy': self._compute_entropy(memberships)
        }
        
        return memberships, aux_info
        
    def _compute_fuzzy_row(self, distances: Tensor) -> Tensor:
        """Compute fuzzy memberships for a single point.
        
        Args:
            distances: (K,) distances to each cluster
            
        Returns:
            (K,) membership values
        """
        power = 2.0 / (self.m - 1.0)
        
        # Compute reciprocal of membership denominators
        # u_k = 1 / Σ_j (d_k / d_j)^power
        inv_distances = 1.0 / distances.clamp(min=self.epsilon)
        
        # For numerical stability, normalize by max inverse distance
        max_inv = inv_distances.max()
        if max_inv > 0:
            normalized_inv = inv_distances / max_inv
            raised = normalized_inv ** power
            denominator = raised.sum()
            
            # Compute final memberships
            memberships = raised / denominator
        else:
            # Fallback: uniform membership
            memberships = torch.ones_like(distances) / len(distances)
            
        return memberships
        
    def _compute_entropy(self, memberships: Tensor) -> float:
        """Compute average entropy of membership distribution.
        
        Higher entropy indicates fuzzier clustering.
        
        Args:
            memberships: (n, K) membership matrix
            
        Returns:
            Average entropy across all points
        """
        # Avoid log(0)
        safe_memberships = memberships.clamp(min=self.epsilon)
        
        # Compute entropy for each point
        entropy = -torch.sum(safe_memberships * torch.log(safe_memberships), dim=1)
        
        return entropy.mean().item()


class PossibilisticAssignment(AssignmentStrategy):
    """Possibilistic assignment for Possibilistic C-Means.
    
    Unlike fuzzy assignment, memberships don't need to sum to 1,
    allowing for outlier detection.
    """
    
    def __init__(self, m: float = 2.0, eta: Optional[Tensor] = None,
                 eta_factor: float = 1.0):
        """
        Args:
            m: Fuzziness exponent (m > 1)
            eta: (K,) typicality parameters for each cluster
            eta_factor: Multiplicative factor for automatic eta estimation
        """
        super().__init__()
        if m <= 1:
            raise ValueError(f"Fuzziness exponent m must be > 1, got {m}")
        self.m = m
        self.eta = eta
        self.eta_factor = eta_factor
        
    @property
    def is_soft(self) -> bool:
        """Possibilistic assignments are soft."""
        return True
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute possibilistic memberships.
        
        Uses the formula:
        t_ik = 1 / (1 + (d_ik / η_k)^(1/(m-1)))
        
        Args:
            points: (n, d) data points
            representations: List of K cluster representations
            
        Returns:
            typicalities: (n, K) typicality matrix
            aux_info: Dictionary with additional information
        """
        n_points = points.shape[0]
        n_clusters = len(representations)
        device = points.device
        
        # Compute distances
        distances = torch.zeros(n_points, n_clusters, device=device)
        for k, representation in enumerate(representations):
            distances[:, k] = representation.distance_to_point(points)
            
        # Estimate eta if not provided
        if self.eta is None:
            # Use average intra-cluster distance as eta
            eta = torch.zeros(n_clusters, device=device)
            
            # First do a hard assignment
            hard_assignments = distances.argmin(dim=1)
            
            for k in range(n_clusters):
                cluster_mask = hard_assignments == k
                if cluster_mask.any():
                    cluster_distances = distances[cluster_mask, k]
                    eta[k] = cluster_distances.mean() * self.eta_factor
                else:
                    # No points assigned - use overall average
                    eta[k] = distances.mean() * self.eta_factor
                    
            # Ensure eta is positive
            eta = eta.clamp(min=1e-6)
        else:
            eta = self.eta.to(device)
            
        # Compute possibilistic memberships
        # t_ik = 1 / (1 + (d_ik / η_k)^(1/(m-1)))
        power = 1.0 / (self.m - 1.0)
        
        normalized_distances = distances / eta.unsqueeze(0).clamp(min=1e-10)
        typicalities = 1.0 / (1.0 + normalized_distances ** power)
        
        # Compute auxiliary information
        aux_info = {
            'distances': distances,
            'eta': eta,
            'outlier_scores': 1.0 - typicalities.max(dim=1)[0],
            'total_typicality': typicalities.sum(dim=1),
            'fuzziness': self.m
        }
        
        return typicalities, aux_info


class FuzzyPossibilisticAssignment(AssignmentStrategy):
    """Combined fuzzy-possibilistic assignment.
    
    Combines fuzzy memberships (sum to 1) with possibilistic typicalities
    for robust clustering with outlier detection.
    """
    
    def __init__(self, m_fuzzy: float = 2.0, m_poss: float = 2.0,
                 a: float = 1.0, b: float = 1.0,
                 eta: Optional[Tensor] = None):
        """
        Args:
            m_fuzzy: Fuzziness exponent for fuzzy part
            m_poss: Fuzziness exponent for possibilistic part
            a: Weight for fuzzy term
            b: Weight for possibilistic term
            eta: Typicality parameters
        """
        super().__init__()
        self.fuzzy_assignment = FuzzyAssignment(m=m_fuzzy)
        self.poss_assignment = PossibilisticAssignment(m=m_poss, eta=eta)
        self.a = a
        self.b = b
        
    @property
    def is_soft(self) -> bool:
        return True
        
    def compute_assignments(self, points: Tensor,
                          representations: List[ClusterRepresentation],
                          **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute combined fuzzy-possibilistic assignments.
        
        Returns:
            Combined assignment matrix and auxiliary information
        """
        # Get fuzzy memberships
        fuzzy_memberships, fuzzy_info = self.fuzzy_assignment.compute_assignments(
            points, representations, **kwargs
        )
        
        # Get possibilistic typicalities
        typicalities, poss_info = self.poss_assignment.compute_assignments(
            points, representations, **kwargs
        )
        
        # Combine: u_ik^a * t_ik^b
        combined = (fuzzy_memberships ** self.a) * (typicalities ** self.b)
        
        # Normalize to sum to 1 (maintaining fuzzy constraint)
        combined = combined / combined.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        # Merge auxiliary information
        aux_info = {
            'fuzzy_memberships': fuzzy_memberships,
            'typicalities': typicalities,
            'distances': fuzzy_info['distances'],
            'eta': poss_info['eta'],
            'outlier_scores': poss_info['outlier_scores'],
            'a': self.a,
            'b': self.b
        }
        
        return combined, aux_info