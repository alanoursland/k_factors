"""
Core data structures for the K-Factors clustering algorithms.

This module provides efficient data structures for storing cluster states,
assignments, and auxiliary information needed by various algorithms.
"""

from typing import Optional, List, Dict, Any, Union
import torch
from torch import Tensor
from dataclasses import dataclass, field
import warnings


@dataclass
class ClusterState:
    """Container for all parameters defining clusters at a given iteration.
    
    This class efficiently stores and manages cluster parameters for all
    clusters simultaneously, enabling batched operations.
    """
    
    # Core parameters present in all algorithms
    means: Tensor  # (K, d) cluster centers/means
    n_clusters: int
    dimension: int
    
    # Optional parameters for different representations
    covariances: Optional[Tensor] = None  # (K, d, d) for GMM
    precisions: Optional[Tensor] = None   # (K, d, d) inverse covariances
    bases: Optional[Tensor] = None        # (K, R, d) orthonormal bases for subspaces
    variances: Optional[Tensor] = None    # (K,) or (K, d) diagonal variances
    mixing_weights: Optional[Tensor] = None  # (K,) mixture weights
    
    # Auxiliary information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate dimensions and set derived attributes."""
        assert self.means.shape == (self.n_clusters, self.dimension)
        
        if self.mixing_weights is None:
            # Default to uniform mixing
            self.mixing_weights = torch.ones(self.n_clusters) / self.n_clusters
            
    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.means.device
    
    def to(self, device: torch.device) -> 'ClusterState':
        """Move all tensors to specified device."""
        def move_if_tensor(x):
            return x.to(device) if isinstance(x, Tensor) else x
            
        return ClusterState(
            means=move_if_tensor(self.means),
            n_clusters=self.n_clusters,
            dimension=self.dimension,
            covariances=move_if_tensor(self.covariances),
            precisions=move_if_tensor(self.precisions),
            bases=move_if_tensor(self.bases),
            variances=move_if_tensor(self.variances),
            mixing_weights=move_if_tensor(self.mixing_weights),
            metadata=self.metadata.copy()
        )
    
    def update_means(self, new_means: Tensor) -> None:
        """Update cluster means in-place."""
        self.means.copy_(new_means)
        
    def get_subspace_basis(self, cluster_idx: int, dim_idx: Optional[int] = None) -> Tensor:
        """Get basis vectors for a specific cluster.
        
        Args:
            cluster_idx: Which cluster
            dim_idx: Optional specific dimension (for sequential methods)
            
        Returns:
            Either (R, d) or (d,) tensor of basis vectors
        """
        if self.bases is None:
            raise ValueError("No basis vectors stored in this ClusterState")
            
        if dim_idx is None:
            return self.bases[cluster_idx]
        else:
            return self.bases[cluster_idx, dim_idx]


class AssignmentMatrix:
    """Efficient storage and manipulation of cluster assignments.
    
    Supports both hard (discrete) and soft (probabilistic) assignments,
    with automatic format conversion and efficient aggregation operations.
    """
    
    def __init__(self, 
                 assignments: Tensor,
                 n_clusters: int,
                 is_soft: bool = False):
        """
        Args:
            assignments: Either (n,) hard assignments or (n, K) soft assignments
            n_clusters: Number of clusters K
            is_soft: Whether assignments are soft (probabilistic)
        """
        self.n_clusters = n_clusters
        self.is_soft = is_soft
        self._validate_and_store(assignments)
        
    def _validate_and_store(self, assignments: Tensor):
        """Validate and store assignments in canonical format."""
        if self.is_soft:
            assert assignments.dim() == 2
            assert assignments.shape[1] == self.n_clusters
            assert torch.allclose(assignments.sum(dim=1), torch.ones(assignments.shape[0]))
            self._soft_assignments = assignments
            self._hard_assignments = None
        else:
            assert assignments.dim() == 1
            assert assignments.max() < self.n_clusters
            assert assignments.min() >= 0
            self._hard_assignments = assignments.long()
            self._soft_assignments = None
            
    @property
    def n_points(self) -> int:
        """Number of data points."""
        if self._hard_assignments is not None:
            return self._hard_assignments.shape[0]
        else:
            return self._soft_assignments.shape[0]
            
    def get_hard(self) -> Tensor:
        """Get hard assignments, converting from soft if necessary."""
        if self._hard_assignments is not None:
            return self._hard_assignments
        else:
            # Convert soft to hard by taking argmax
            return self._soft_assignments.argmax(dim=1)
            
    def get_soft(self) -> Tensor:
        """Get soft assignments, converting from hard if necessary."""
        if self._soft_assignments is not None:
            return self._soft_assignments
        else:
            # Convert hard to soft using one-hot encoding
            soft = torch.zeros(self.n_points, self.n_clusters, 
                              device=self._hard_assignments.device)
            soft.scatter_(1, self._hard_assignments.unsqueeze(1), 1.0)
            return soft
            
    def get_cluster_indices(self, cluster_idx: int) -> Tensor:
        """Get indices of points assigned to a specific cluster."""
        if self._hard_assignments is not None:
            return torch.where(self._hard_assignments == cluster_idx)[0]
        else:
            # For soft assignments, threshold at 0.5 or take points with max responsibility
            return torch.where(self._soft_assignments[:, cluster_idx] > 0.5)[0]
            
    def get_cluster_weights(self, cluster_idx: int) -> Tensor:
        """Get weights for points in a cluster (1.0 for hard, responsibilities for soft)."""
        if self._hard_assignments is not None:
            indices = self.get_cluster_indices(cluster_idx)
            return torch.ones(len(indices), device=self._hard_assignments.device)
        else:
            return self._soft_assignments[:, cluster_idx]
            
    def count_per_cluster(self) -> Tensor:
        """Count points per cluster (weighted for soft assignments)."""
        if self._hard_assignments is not None:
            return torch.bincount(self._hard_assignments, minlength=self.n_clusters).float()
        else:
            return self._soft_assignments.sum(dim=0)
            
    def to(self, device: torch.device) -> 'AssignmentMatrix':
        """Move to specified device."""
        if self._hard_assignments is not None:
            new_assignments = self._hard_assignments.to(device)
        else:
            new_assignments = self._soft_assignments.to(device)
        return AssignmentMatrix(new_assignments, self.n_clusters, self.is_soft)


class DirectionTracker:
    """Tracks claimed directions for K-Factors algorithm.
    
    Efficiently stores which basis directions each point has claimed
    across iterations, enabling penalty computation.
    """
    
    def __init__(self, n_points: int, n_clusters: int, device: torch.device):
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.device = device
        
        # List of claimed directions per point
        self.claimed_directions: List[List[Tensor]] = [[] for _ in range(n_points)]
        
        # Current iteration/stage
        self.current_stage = 0
        
    def add_claimed_direction(self, point_idx: int, cluster_idx: int, 
                            direction: Tensor) -> None:
        """Record that a point claimed a specific direction."""
        self.claimed_directions[point_idx].append(direction.detach())
        
    def add_claimed_directions_batch(self, assignments: Tensor, 
                                   cluster_bases: Tensor) -> None:
        """Batch update of claimed directions for current stage.
        
        Args:
            assignments: (n,) hard cluster assignments
            cluster_bases: (K, R, d) current basis vectors, or (K, d) for single direction
        """
        for i in range(self.n_points):
            cluster = assignments[i].item()
            if cluster_bases.dim() == 3:
                # Multiple directions per cluster - take current stage
                direction = cluster_bases[cluster, self.current_stage]
            else:
                # Single direction per cluster
                direction = cluster_bases[cluster]
            self.claimed_directions[i].append(direction.detach())
            
    def compute_penalty(self, point_idx: int, test_direction: Tensor, 
                       penalty_type: str = 'product') -> float:
        """Compute penalty for using a direction based on claimed history.
        
        Args:
            point_idx: Which point
            test_direction: Direction to test
            penalty_type: 'product' or 'sum' penalty
            
        Returns:
            Penalty value in [0, 1]
        """
        if not self.claimed_directions[point_idx]:
            return 1.0  # No penalty if no claimed directions
            
        similarities = []
        for claimed_dir in self.claimed_directions[point_idx]:
            # Compute absolute cosine similarity
            sim = torch.abs(torch.dot(test_direction, claimed_dir))
            similarities.append(sim)
            
        similarities = torch.stack(similarities)
        
        if penalty_type == 'product':
            # Product of (1 - similarity) terms
            penalty = torch.prod(1.0 - similarities)
        elif penalty_type == 'sum':
            # Average of (1 - similarity) terms
            penalty = torch.mean(1.0 - similarities)
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")
            
        return penalty.item()
        
    def compute_penalty_batch(self, test_directions: Tensor, 
                            penalty_type: str = 'product') -> Tensor:
        """Compute penalties for multiple points and directions.
        
        Args:
            test_directions: (n, K, d) directions to test for each point-cluster pair
            penalty_type: 'product' or 'sum' penalty
            
        Returns:
            (n, K) penalty matrix
        """
        n, k, d = test_directions.shape
        penalties = torch.ones(n, k, device=self.device)
        
        for i in range(n):
            if not self.claimed_directions[i]:
                continue  # No penalty
                
            claimed_stack = torch.stack(self.claimed_directions[i])  # (history, d)
            
            for j in range(k):
                # Compute similarities to all claimed directions
                similarities = torch.abs(
                    torch.matmul(claimed_stack, test_directions[i, j])
                )  # (history,)
                
                if penalty_type == 'product':
                    penalties[i, j] = torch.prod(1.0 - similarities)
                elif penalty_type == 'sum':
                    penalties[i, j] = torch.mean(1.0 - similarities)
                    
        return penalties
        
    def advance_stage(self) -> None:
        """Move to next stage/iteration."""
        self.current_stage += 1
        
    def reset(self) -> None:
        """Clear all claimed directions."""
        self.claimed_directions = [[] for _ in range(self.n_points)]
        self.current_stage = 0
        
    def to(self, device: torch.device) -> 'DirectionTracker':
        """Move to specified device."""
        new_tracker = DirectionTracker(self.n_points, self.n_clusters, device)
        new_tracker.current_stage = self.current_stage
        for i in range(self.n_points):
            new_tracker.claimed_directions[i] = [
                d.to(device) for d in self.claimed_directions[i]
            ]
        return new_tracker


@dataclass 
class AlgorithmState:
    """Complete state of a clustering algorithm at a given iteration.
    
    Used for checkpointing, convergence checking, and debugging.
    """
    iteration: int
    cluster_state: ClusterState
    assignments: AssignmentMatrix
    objective_value: float
    
    # Optional components
    direction_tracker: Optional[DirectionTracker] = None
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: torch.device) -> 'AlgorithmState':
        """Move all components to specified device."""
        return AlgorithmState(
            iteration=self.iteration,
            cluster_state=self.cluster_state.to(device),
            assignments=self.assignments.to(device),
            objective_value=self.objective_value,
            direction_tracker=self.direction_tracker.to(device) if self.direction_tracker else None,
            converged=self.converged,
            metadata=self.metadata.copy()
        )