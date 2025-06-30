"""
K-Lines clustering algorithm.

Special case of K-Subspaces where each cluster is represented by a 1-dimensional
line (plus a mean). Useful for data that lies approximately along different
directions.
"""

from typing import Optional, List
import torch
from torch import Tensor

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import (
    ClusterRepresentation, ClusteringObjective,
    InitializationStrategy, ConvergenceCriterion
)
from ..representations.subspace import SubspaceRepresentation
from ..assignments.hard import HardAssignment
from ..distances.subspace import OrthogonalDistance
from ..updates.pca import PCAUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective


class KLinesObjective(ClusteringObjective):
    """K-Lines objective: sum of squared orthogonal distances to lines."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor) -> Tensor:
        """Compute total orthogonal distance."""
        total = 0.0
        distance_metric = OrthogonalDistance()
        
        for k, rep in enumerate(representations):
            cluster_mask = (assignments == k)
            if cluster_mask.any():
                cluster_points = points[cluster_mask]
                distances = distance_metric.compute(cluster_points, rep)
                total += distances.sum()
                
        return total
        
    @property
    def minimize(self) -> bool:
        return True


class KLines(BaseClusteringAlgorithm):
    """K-Lines clustering algorithm.
    
    Clusters data by fitting K one-dimensional lines (affine subspaces).
    Each cluster is represented by a mean point and a direction vector.
    Points are assigned to the line with minimum orthogonal distance.
    
    Parameters
    ----------
    n_clusters : int
        Number of lines/clusters
    init : str, default='k-means++'
        Initialization method
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed
    device : torch.device, optional
        Computation device
        
    Attributes
    ----------
    cluster_centers_ : Tensor of shape (n_clusters, n_features)
        Cluster centroids (points on the lines)
    cluster_directions_ : Tensor of shape (n_clusters, n_features)
        Unit direction vectors for each line
    labels_ : Tensor of shape (n_samples,)
        Cluster assignments
    inertia_ : float
        Sum of squared orthogonal distances
    n_iter_ : int
        Number of iterations run
    
    Examples
    --------
    >>> import torch
    >>> from kfactors.algorithms import KLines
    >>> 
    >>> # Generate data along different lines
    >>> X = torch.randn(300, 10)
    >>> 
    >>> # Fit K-Lines
    >>> klines = KLines(n_clusters=3, verbose=1)
    >>> klines.fit(X)
    >>> 
    >>> # Get cluster assignments
    >>> labels = klines.labels_
    >>> 
    >>> # Project new points onto their assigned lines
    >>> X_new = torch.randn(10, 10)
    >>> labels_new = klines.predict(X_new)
    """
    
    def __init__(self,
                 n_clusters: int,
                 init: str = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize K-Lines algorithm."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        self.init = init
        
        # K-Lines specific attributes
        self.cluster_directions_ = None
        self.labels_ = None
        
    def _create_components(self) -> None:
        """Create K-Lines specific components."""
        # Hard assignment based on orthogonal distance
        self.assignment_strategy = HardAssignment()
        
        # PCA updater extracts first principal component
        self.update_strategy = PCAUpdater(n_components=1)
        
        # Initialization
        if self.init == 'k-means++':
            self.initialization_strategy = KMeansPlusPlusInit()
        else:
            from ..initialization.random import RandomInit
            self.initialization_strategy = RandomInit()
            
        # Convergence based on objective change
        self.convergence_criterion = ChangeInObjective(rel_tol=self.tol)
        
        # Objective
        self.objective = KLinesObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create 1D subspace representations."""
        representations = []
        dimension = data.shape[1]
        
        # Get initial centers
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        # Convert to 1D subspace representations
        for init_rep in init_representations:
            subspace = SubspaceRepresentation(
                dimension=dimension,
                subspace_dim=1,  # Lines are 1D subspaces
                device=self.device
            )
            
            # Set initial mean
            subspace.mean = init_rep.get_parameters()['mean']
            
            representations.append(subspace)
            
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'KLines':
        """Fit K-Lines clustering.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : KLines
            Fitted estimator
        """
        super().fit(X, y)
        
        # Extract final results
        if self.history_:
            final_state = self.history_[-1]
            self.labels_ = final_state.assignments.get_hard()
            
            # Extract direction vectors
            self.cluster_directions_ = torch.zeros(
                self.n_clusters, X.shape[1], device=self.device
            )
            for k, rep in enumerate(self.representations):
                params = rep.get_parameters()
                if params['basis'].shape[1] > 0:
                    self.cluster_directions_[k] = params['basis'][:, 0]
                    
        return self
        
    def project_to_lines(self, X: Tensor) -> Tensor:
        """Project points onto their assigned lines.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Points to project
            
        Returns
        -------
        X_projected : Tensor of shape (n_samples, n_features)
            Projected points
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before projection")
            
        X = self._validate_data(X)
        labels = self.predict(X)
        
        X_projected = torch.zeros_like(X)
        
        for k, rep in enumerate(self.representations):
            mask = (labels == k)
            if mask.any():
                cluster_points = X[mask]
                _, projections = rep.project_points(cluster_points)
                X_projected[mask] = projections
                
        return X_projected
        
    def get_line_parameters(self, cluster_idx: int) -> Tuple[Tensor, Tensor]:
        """Get parameters defining a cluster's line.
        
        Parameters
        ----------
        cluster_idx : int
            Cluster index
            
        Returns
        -------
        point : Tensor of shape (n_features,)
            A point on the line (cluster mean)
        direction : Tensor of shape (n_features,)
            Unit direction vector of the line
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        if cluster_idx < 0 or cluster_idx >= self.n_clusters:
            raise ValueError(f"Invalid cluster index: {cluster_idx}")
            
        point = self.cluster_centers_[cluster_idx]
        direction = self.cluster_directions_[cluster_idx]
        
        return point, direction