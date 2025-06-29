"""
K-means clustering algorithm.

The classic K-means algorithm implemented using the modular framework.
"""

from typing import Optional, List
import torch
from torch import Tensor

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import (
    ClusterRepresentation, AssignmentStrategy, ParameterUpdater,
    InitializationStrategy, ConvergenceCriterion, ClusteringObjective
)
from ..representations.centroid import CentroidRepresentation
from ..assignments.hard import HardAssignment
from ..distances.euclidean import EuclideanDistance
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInAssignments
from ..updates.mean import MeanUpdater


class KMeansObjective(ClusteringObjective):
    """K-means objective: sum of squared distances to centroids."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor) -> Tensor:
        """Compute within-cluster sum of squares."""
        total = 0.0
        
        # For hard assignments
        for k, rep in enumerate(representations):
            cluster_points_mask = (assignments == k)
            if cluster_points_mask.any():
                cluster_points = points[cluster_points_mask]
                distances = rep.distance_to_point(cluster_points)
                total += distances.sum()
                
        return total
        
    @property
    def minimize(self) -> bool:
        return True


class KMeans(BaseClusteringAlgorithm):
    """K-means clustering algorithm.
    
    Classic K-means that partitions data into K clusters by minimizing
    within-cluster sum of squared distances.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    init : str or array-like, default='k-means++'
        Initialization method:
        - 'k-means++' : K-means++ initialization
        - 'random' : Random initialization
        - array of shape (n_clusters, n_features) : Use as initial centers
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance based on change in assignments
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed for reproducibility
    device : torch.device, optional
        Device for computation (CPU/GPU)
        
    Attributes
    ----------
    cluster_centers_ : Tensor of shape (n_clusters, n_features)
        Cluster centroids
    labels_ : Tensor of shape (n_samples,)
        Cluster assignments for training data
    inertia_ : float
        Sum of squared distances to nearest cluster center
    n_iter_ : int
        Number of iterations run
    """
    
    def __init__(self,
                 n_clusters: int,
                 init: str = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize K-means algorithm."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        self.init = init
        self._initial_centers = None
        
        # Store labels for sklearn compatibility
        self.labels_ = None
        
    def _create_components(self) -> None:
        """Create K-means specific components."""
        # Assignment strategy
        self.assignment_strategy = HardAssignment()
        
        # Update strategy
        self.update_strategy = MeanUpdater()
        
        # Initialization
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            elif self.init == 'random':
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
            else:
                raise ValueError(f"Unknown init method: {self.init}")
        else:
            # Custom initial centers provided
            self._initial_centers = torch.tensor(self.init, device=self.device)
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self._initial_centers)
            
        # Convergence criterion
        self.convergence_criterion = ChangeInAssignments(min_change_fraction=self.tol)
        
        # Objective
        self.objective = KMeansObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create centroid representations."""
        representations = []
        dimension = data.shape[1]
        
        # Get initial centers
        initial_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        # Convert to CentroidRepresentation if needed
        for init_rep in initial_representations:
            if isinstance(init_rep, CentroidRepresentation):
                representations.append(init_rep)
            else:
                # Create new centroid representation
                centroid = CentroidRepresentation(dimension, self.device)
                # Copy mean from initialization
                centroid.mean = init_rep.get_parameters()['mean']
                representations.append(centroid)
                
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'KMeans':
        """Fit K-means clustering.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : KMeans
            Fitted estimator
        """
        super().fit(X, y)
        
        # Store final labels
        if self.history_:
            self.labels_ = self.history_[-1].assignments.get_hard()
            
        return self
        
    def fit_predict(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Fit and return labels.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        labels : Tensor of shape (n_samples,)
            Cluster labels
        """
        self.fit(X, y)
        return self.labels_
        
    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels for new data.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            New data to predict
            
        Returns
        -------
        labels : Tensor of shape (n_samples,)
            Cluster labels
        """
        return super().predict(X)
        
    def score(self, X: Tensor, y: Optional[Tensor] = None) -> float:
        """Opposite of the value of X on the K-means objective.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            New data
        y : Ignored
            Not used
            
        Returns
        -------
        score : float
            Negative of sum of squared distances to centers
        """
        labels = self.predict(X)
        distances = 0.0
        
        for k, rep in enumerate(self.representations):
            cluster_mask = (labels == k)
            if cluster_mask.any():
                cluster_points = X[cluster_mask]
                distances += rep.distance_to_point(cluster_points).sum().item()
                
        return -distances