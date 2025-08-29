"""
Core interfaces for the K-Factors family of clustering algorithms.

This module defines the abstract base classes that all components must implement,
ensuring a consistent API across different clustering methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union
import torch
from torch import Tensor


class ClusterRepresentation(ABC):
    """Abstract base class for cluster representations.
    
    Different clustering algorithms represent clusters differently:
    - K-means: just centroids
    - K-subspaces: centroid + subspace basis
    - GMM: mean + covariance
    - K-factors: mean + sequential orthonormal basis
    """
    
    @abstractmethod
    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """Compute distance/cost from points to this cluster representation.
        
        Args:
            points: (n, d) tensor of data points
            indices: Optional (n,) tensor of point indices for tracking
            
        Returns:
            (n,) tensor of distances/costs
        """
        pass
    
    @abstractmethod
    def update_from_points(self, points: Tensor, weights: Optional[Tensor] = None, 
                          **kwargs) -> None:
        """Update cluster parameters given assigned points.
        
        Args:
            points: (n, d) tensor of assigned points
            weights: Optional (n,) tensor of weights for soft assignments
            **kwargs: Additional update-specific parameters
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Tensor]:
        """Return all parameters defining this cluster representation."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set cluster parameters from dictionary."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Ambient dimension of the data."""
        pass
    
    @abstractmethod
    def to(self, device: torch.device) -> 'ClusterRepresentation':
        """Move representation to specified device."""
        pass


class AssignmentStrategy(ABC):
    """Abstract base class for point-to-cluster assignment strategies."""
    
    @abstractmethod
    def compute_assignments(self, points: Tensor, 
                          representations: list[ClusterRepresentation],
                          **kwargs) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Compute cluster assignments for points.
        
        Args:
            points: (n, d) tensor of data points
            representations: List of K cluster representations
            **kwargs: Strategy-specific parameters
            
        Returns:
            Either:
            - (n,) tensor of hard assignments (cluster indices)
            - (n, K) tensor of soft assignments (responsibilities)
            - (n, K) tensor of feature indices ordered by distance
              aux_info: {'distances': (n, K) corresponding distances}
            - Tuple of (assignments, auxiliary_info) for strategies that track state
        """
        pass
    
    @property
    @abstractmethod
    def is_soft(self) -> bool:
        """Whether this strategy produces soft (probabilistic) assignments."""
        return False

    @property
    def is_ranked(self) -> bool:
        return False

class ParameterUpdater(ABC):
    """Abstract base class for cluster parameter update strategies."""
    
    @abstractmethod
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignment_weights: Tensor,
               **kwargs) -> None:
        """Update cluster parameters given points and assignments.
        
        Args:
            representation: Cluster representation to update
            points: (n, d) tensor of all data points
            assignment_weights: Either (n,) hard or (n, K) soft assignments
            **kwargs: Update-specific parameters
        """
        pass


class DistanceMetric(ABC):
    """Abstract base class for distance/cost computations."""
    
    @abstractmethod
    def compute(self, points: Tensor, representation: ClusterRepresentation,
                **kwargs) -> Tensor:
        """Compute distances from points to cluster.
        
        Args:
            points: (n, d) tensor of points
            representation: Cluster representation
            **kwargs: Metric-specific parameters
            
        Returns:
            (n,) tensor of distances/costs
        """
        pass


class InitializationStrategy(ABC):
    """Abstract base class for cluster initialization strategies."""
    
    @abstractmethod
    def initialize(self, points: Tensor, n_clusters: int, 
                  **kwargs) -> list[ClusterRepresentation]:
        """Initialize cluster representations.
        
        Args:
            points: (n, d) tensor of data points
            n_clusters: Number of clusters to initialize
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of initialized cluster representations
        """
        pass


class ConvergenceCriterion(ABC):
    """Abstract base class for convergence checking."""
    
    def __init__(self):
        self.history = []
    
    @abstractmethod
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Check if algorithm has converged.
        
        Args:
            current_state: Dictionary containing current algorithm state
            
        Returns:
            True if converged, False otherwise
        """
        pass
    
    def reset(self):
        """Reset convergence history."""
        self.history = []


class ClusteringObjective(ABC):
    """Abstract base class for clustering objective functions."""
    
    @abstractmethod
    def compute(self, points: Tensor, 
                representations: list[ClusterRepresentation],
                assignments: Tensor) -> Tensor:
        """Compute objective function value.
        
        Args:
            points: (n, d) tensor of data points
            representations: List of cluster representations
            assignments: Cluster assignments (hard or soft)
            
        Returns:
            Scalar objective value
        """
        pass
    
    @property
    @abstractmethod
    def minimize(self) -> bool:
        """Whether to minimize (True) or maximize (False) this objective."""
        pass