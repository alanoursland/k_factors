"""
K-Subspaces clustering algorithm.

General subspace clustering where each cluster is represented by an
r-dimensional affine subspace. K-Lines is a special case with r=1.
"""

from typing import Optional, List, Union
import torch
from torch import Tensor

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import ClusterRepresentation, ClusteringObjective
from ..representations.subspace import SubspaceRepresentation
from ..assignments.hard import HardAssignment
from ..distances.subspace import OrthogonalDistance
from ..updates.pca import PCAUpdater, IncrementalPCAUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective, CombinedCriterion, ChangeInAssignments


class KSubspacesObjective(ClusteringObjective):
    """K-Subspaces objective: sum of squared orthogonal distances."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor) -> Tensor:
        """Compute total orthogonal distance to subspaces."""
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


class KSubspaces(BaseClusteringAlgorithm):
    """K-Subspaces clustering algorithm.
    
    Partitions data into K clusters, each represented by an r-dimensional
    affine subspace. Points are assigned to the subspace with minimum
    orthogonal distance.
    
    This is a generalization of K-Lines (r=1) and approaches K-means
    as r→0. When r=d-1 (where d is data dimension), each cluster can
    perfectly fit any configuration of points.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters/subspaces
    n_components : int
        Dimension of each subspace (r)
    init : str or array-like, default='k-means++'
        Initialization method:
        - 'k-means++': Use K-means++ for initial centers
        - 'random': Random point selection
        - array: Custom initial centers
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
    algorithm : str, default='full'
        PCA algorithm:
        - 'full': Standard batch PCA
        - 'incremental': Incremental PCA for large datasets
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed
    device : torch.device, optional
        Computation device
        
    Attributes
    ----------
    cluster_centers_ : Tensor of shape (n_clusters, n_features)
        Cluster centroids
    cluster_subspaces_ : Tensor of shape (n_clusters, n_features, n_components)
        Orthonormal basis for each cluster's subspace
    labels_ : Tensor of shape (n_samples,)
        Cluster assignments
    inertia_ : float
        Sum of squared distances to nearest subspace
    n_iter_ : int
        Number of iterations run
        
    Examples
    --------
    >>> import torch
    >>> from kfactors.algorithms import KSubspaces
    >>> 
    >>> # Data with intrinsic 2D structure in 10D space
    >>> X = torch.randn(500, 10)
    >>> 
    >>> # Fit with 2D subspaces
    >>> ksub = KSubspaces(n_clusters=3, n_components=2)
    >>> ksub.fit(X)
    >>> 
    >>> # Transform to local coordinates
    >>> X_local = ksub.transform(X)
    >>> 
    >>> # Reconstruct from local coordinates
    >>> X_recon = ksub.inverse_transform(X_local)
    """
    
    def __init__(self,
                 n_clusters: int,
                 n_components: int,
                 init: Union[str, Tensor] = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 algorithm: str = 'full',
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize K-Subspaces algorithm."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.n_components = n_components
        self.init = init
        self.algorithm = algorithm
        
        # K-Subspaces specific attributes
        self.cluster_subspaces_ = None
        self.labels_ = None
        
    def _create_components(self) -> None:
        """Create K-Subspaces specific components."""
        # Hard assignment based on orthogonal distance
        self.assignment_strategy = HardAssignment()
        
        # PCA updater
        if self.algorithm == 'full':
            self.update_strategy = PCAUpdater(n_components=self.n_components)
        elif self.algorithm == 'incremental':
            self.update_strategy = IncrementalPCAUpdater(
                n_components=self.n_components,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
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
            # Custom initialization
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        # Combined convergence criterion
        self.convergence_criterion = CombinedCriterion([
            ChangeInObjective(rel_tol=self.tol),
            ChangeInAssignments(min_change_fraction=0.001)
        ], mode='any')
        
        # Objective
        self.objective = KSubspacesObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create subspace representations."""
        representations = []
        dimension = data.shape[1]
        
        # Validate subspace dimension
        if self.n_components > dimension:
            raise ValueError(f"Subspace dimension {self.n_components} cannot exceed "
                           f"data dimension {dimension}")
                           
        if self.n_components < 0:
            raise ValueError(f"Subspace dimension must be non-negative, got {self.n_components}")
            
        # Get initial centers
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        # Convert to subspace representations
        for init_rep in init_representations:
            subspace = SubspaceRepresentation(
                dimension=dimension,
                subspace_dim=self.n_components,
                device=self.device
            )
            
            # Set initial mean
            subspace.mean = init_rep.get_parameters()['mean']
            
            representations.append(subspace)
            
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'KSubspaces':
        """Fit K-Subspaces clustering.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : KSubspaces
            Fitted estimator
        """
        if self.verbose:
            print(f"K-Subspaces: {self.n_clusters} clusters with "
                  f"{self.n_components}-dimensional subspaces")
                  
        super().fit(X, y)
        
        # Extract final results
        if self.history_:
            final_state = self.history_[-1]
            self.labels_ = final_state.assignments.get_hard()
            
            # Extract subspace bases
            self.cluster_subspaces_ = torch.zeros(
                self.n_clusters, X.shape[1], self.n_components,
                device=self.device
            )
            
            for k, rep in enumerate(self.representations):
                params = rep.get_parameters()
                basis = params['basis']
                if basis.shape[1] > 0:
                    self.cluster_subspaces_[k, :, :basis.shape[1]] = basis
                    
        return self
        
    def transform(self, X: Tensor) -> Tensor:
        """Transform data to cluster-local subspace coordinates.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : Tensor of shape (n_samples, n_components)
            Local subspace coordinates
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling transform")
            
        X = self._validate_data(X)
        labels = self.predict(X)
        
        X_transformed = torch.zeros(len(X), self.n_components, device=self.device)
        
        for k, rep in enumerate(self.representations):
            mask = (labels == k)
            if mask.any():
                cluster_points = X[mask]
                coeffs, _ = rep.project_points(cluster_points)
                X_transformed[mask] = coeffs
                
        return X_transformed
        
    def inverse_transform(self, X_transformed: Tensor, 
                         labels: Optional[Tensor] = None) -> Tensor:
        """Transform from subspace coordinates back to original space.
        
        Parameters
        ----------
        X_transformed : Tensor of shape (n_samples, n_components)
            Local subspace coordinates
        labels : Tensor of shape (n_samples,), optional
            Cluster assignments (will be inferred if not provided)
            
        Returns
        -------
        X : Tensor of shape (n_samples, n_features)
            Reconstructed data
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        n_samples = X_transformed.shape[0]
        d = self.cluster_centers_.shape[1]
        
        # Infer labels if not provided
        if labels is None:
            # Assign to nearest cluster by centroid
            # (This is approximate - better to provide labels)
            fake_points = self.cluster_centers_[0].unsqueeze(0).expand(n_samples, -1)
            labels = self.predict(fake_points)
            
        X_reconstructed = torch.zeros(n_samples, d, device=self.device)
        
        for k, rep in enumerate(self.representations):
            mask = (labels == k)
            if mask.any():
                local_coords = X_transformed[mask]
                params = rep.get_parameters()
                
                # Reconstruct: x = mean + basis @ coords
                X_reconstructed[mask] = (
                    params['mean'].unsqueeze(0) +
                    torch.matmul(local_coords, params['basis'].t())
                )
                
        return X_reconstructed
        
    def reconstruction_error(self, X: Tensor) -> float:
        """Compute reconstruction error.
        
        Parameters
        ----------
        X : Tensor
            Data points
            
        Returns
        -------
        error : float
            Mean squared reconstruction error
        """
        X_trans = self.transform(X)
        X_recon = self.inverse_transform(X_trans, self.predict(X))
        return torch.mean((X - X_recon) ** 2).item()