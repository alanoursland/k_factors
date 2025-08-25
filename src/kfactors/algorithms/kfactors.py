"""
K-Factors clustering algorithm.

The main algorithm that performs sequential extraction of principal components
with a penalty mechanism to ensure diverse subspace representations.
"""

from typing import Optional, List, Dict, Any
import torch
from torch import Tensor
import warnings

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import (
    ClusterRepresentation, ClusteringObjective,
    InitializationStrategy, ConvergenceCriterion
)
from ..base.data_structures import DirectionTracker, AssignmentMatrix, ClusterState
from ..representations.ppca import PPCARepresentation
from ..assignments.penalized import PenalizedAssignment
from ..updates.sequential_pca import SequentialPCAUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective


class KFactorsObjective(ClusteringObjective):
    """K-Factors objective: penalized reconstruction error."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor, **kwargs) -> Tensor:
        """Compute sum of squared residuals after projection."""
        total_error = 0.0
        
        # Extract auxiliary info if available
        aux_info = kwargs.get('aux_info', {})
        distances = aux_info.get('distances', None)
        
        if distances is not None:
            # Use precomputed distances from assignment step
            for k in range(len(representations)):
                cluster_mask = (assignments == k)
                if cluster_mask.any():
                    total_error += distances[cluster_mask, k].sum()
        else:
            # Compute distances directly
            for k, rep in enumerate(representations):
                cluster_mask = (assignments == k)
                if cluster_mask.any():
                    cluster_points = points[cluster_mask]
                    total_error += rep.distance_to_point(cluster_points).sum()
                    
        return total_error
        
    @property
    def minimize(self) -> bool:
        return True


class KFactors(BaseClusteringAlgorithm):
    """K-Factors clustering with sequential subspace extraction.
    
    K-Factors extends K-subspaces by extracting basis vectors sequentially
    with a penalty mechanism that prevents points from repeatedly using
    the same directions. This encourages diverse subspace usage.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    n_components : int
        Number of principal components per cluster (R in the paper)
    init : str or array-like, default='k-means++'
        Initialization method
    penalty_type : str, default='product'
        Type of penalty: 'product' or 'sum'
    penalty_weight : float, default=0.9
        Weight for penalty term (0=no penalty, 1=full penalty)
    max_iter : int, default=30
        Maximum iterations per stage
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
        Cluster centroids
    cluster_bases_ : Tensor of shape (n_clusters, n_components, n_features)
        Orthonormal basis vectors for each cluster
    labels_ : Tensor of shape (n_samples,)
        Final cluster assignments
    direction_tracker_ : DirectionTracker
        Tracks claimed directions per point
    n_iter_ : int
        Total iterations across all stages
    """
    
    def __init__(self,
                 n_clusters: int,
                 n_components: int,
                 init: str = 'k-means++',
                 penalty_type: str = 'product',
                 penalty_weight: float = 0.9,
                 max_iter: int = 30,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize K-Factors algorithm."""
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
        self.penalty_type = penalty_type
        self.penalty_weight = penalty_weight
        
        # K-Factors specific attributes
        self.direction_tracker_ = None
        self.cluster_bases_ = None
        self.stage_history_ = []
        
    def _create_components(self) -> None:
        """Create K-Factors specific components."""
        # Assignment with penalty
        self.assignment_strategy = PenalizedAssignment(
            penalty_type=self.penalty_type,
            penalty_weight=self.penalty_weight
        )
        
        # Sequential PCA updater
        self.update_strategy = SequentialPCAUpdater()
        
        # Initialization
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            else:
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
        else:
            # Custom initialization
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        # Convergence based on objective change
        self.convergence_criterion = ChangeInObjective(
            rel_tol=self.tol,
            patience=2
        )
        
        # Objective
        self.objective = KFactorsObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create PPCA representations for K-Factors."""
        n_points, dimension = data.shape
        representations = []
        
        # Initialize direction tracker
        self.direction_tracker_ = DirectionTracker(
            n_points, self.n_clusters, self.device
        )
        
        # Get initial centers
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        # Convert to PPCA representations
        for init_rep in init_representations:
            ppca = PPCARepresentation(
                dimension=dimension,
                latent_dim=self.n_components,
                device=self.device,
                init_variance=1.0
            )
            
            # Set initial mean
            ppca.mean = init_rep.get_parameters()['mean']
            
            representations.append(ppca)
            
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'KFactors':
        """Fit K-Factors clustering.
        
        Performs sequential extraction of principal components with
        penalty-based assignment.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : KFactors
            Fitted estimator
        """
        # Validate data
        X = self._validate_data(X)
        n_points, dimension = X.shape
        
        if self.verbose:
            print(f"K-Factors: {self.n_clusters} clusters, "
                  f"{self.n_components} components per cluster")
            
        # Create components
        self._create_components()
        
        # Initialize representations and direction tracker
        self.representations = self._create_representations(X)
        
        # Main sequential extraction loop
        self.n_iter_ = 0
        self.stage_history_ = []
        
        for stage in range(self.n_components):
            if self.verbose:
                print(f"\n=== Stage {stage + 1}/{self.n_components} ===")
                
            # Reset convergence for this stage
            self.convergence_criterion.reset()
            stage_converged = False
            
            # Update current stage in direction tracker
            self.direction_tracker_.current_stage = stage
            
            # Inner loop for this stage
            for iter_in_stage in range(self.max_iter):
                # Assignment step with penalty
                assignments, aux_info = self.assignment_strategy.compute_assignments(
                    X,
                    self.representations,
                    direction_tracker=self.direction_tracker_,
                    current_stage=stage
                )
                
                assignment_matrix = AssignmentMatrix(
                    assignments, self.n_clusters, is_soft=False
                )
                
                # Update each cluster's representation
                for k, representation in enumerate(self.representations):
                    cluster_indices = assignment_matrix.get_cluster_indices(k)
                    
                    if len(cluster_indices) > 0:
                        cluster_points = X[cluster_indices]
                        self.update_strategy.update(
                            representation,
                            cluster_points,
                            current_stage=stage
                        )
                        
                # Compute objective
                objective_value = self.objective.compute(
                    X, self.representations, assignments, aux_info=aux_info
                )
                
                # Check convergence for this stage
                cluster_state = self._extract_cluster_state()
                stage_converged = self.convergence_criterion.check({
                    'iteration': iter_in_stage,
                    'objective': objective_value.item(),
                    'assignments': assignments,
                    'cluster_state': cluster_state
                })
                
                if self.verbose >= 2:
                    print(f"  Iteration {iter_in_stage:3d}: "
                          f"objective = {objective_value.item():.6f}")
                          
                if stage_converged:
                    if self.verbose:
                        print(f"  Stage {stage + 1} converged at iteration {iter_in_stage}")
                    break
                    
                self.n_iter_ += 1
                
            # Update direction tracker with claimed directions
            self.direction_tracker_.add_claimed_directions_batch(
                assignments,
                torch.stack([rep.W[:, stage] for rep in self.representations])
            )
            
            # Store stage results
            self.stage_history_.append({
                'stage': stage,
                'iterations': iter_in_stage + 1,
                'objective': objective_value.item(),
                'converged': stage_converged
            })
            
            if not stage_converged and self.verbose:
                warnings.warn(f"Stage {stage + 1} did not converge")
                
        # Store final results
        self.labels_ = assignments
        self.cluster_centers_ = cluster_state.means
        self.cluster_bases_ = torch.stack([
            rep.W for rep in self.representations
        ])
        
        self.fitted_ = True
        
        if self.verbose:
            print(f"\nK-Factors completed: {self.n_iter_} total iterations")
            
        return self
        
    def transform(self, X: Tensor) -> Tensor:
        """Transform data to cluster-local coordinates.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : Tensor of shape (n_samples, n_clusters * n_components)
            Transformed data (concatenated local coordinates)
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling transform")
            
        X = self._validate_data(X)
        n_points = X.shape[0]
        
        # Get cluster assignments
        assignments = self.predict(X)
        
        # Transform each point using its cluster's basis
        transformed = torch.zeros(
            n_points, self.n_clusters * self.n_components,
            device=self.device
        )
        
        for k, rep in enumerate(self.representations):
            cluster_mask = (assignments == k)
            if cluster_mask.any():
                cluster_points = X[cluster_mask]
                centered = cluster_points - rep.mean.unsqueeze(0)
                
                # Project onto cluster basis
                local_coords = torch.matmul(centered, rep.W)
                
                # Store in output
                start_idx = k * self.n_components
                end_idx = (k + 1) * self.n_components
                transformed[cluster_mask, start_idx:end_idx] = local_coords
                
        return transformed
        
    def inverse_transform(self, X_transformed: Tensor, assignments: Optional[Tensor] = None) -> Tensor:
        """Transform from cluster-local coordinates back to original space.
        
        Parameters
        ----------
        X_transformed : Tensor
            Transformed data from transform()
        assignments : Tensor, optional
            Cluster assignments (will be inferred if not provided)
            
        Returns
        -------
        X : Tensor of shape (n_samples, n_features)
            Data in original space
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        n_points = X_transformed.shape[0]
        reconstructed = torch.zeros(n_points, self.cluster_centers_.shape[1],
                                  device=self.device)
                                  
        # Infer assignments if not provided
        if assignments is None:
            # Check which cluster's coordinates are non-zero
            assignments = torch.zeros(n_points, dtype=torch.long, device=self.device)
            for i in range(n_points):
                for k in range(self.n_clusters):
                    start_idx = k * self.n_components
                    end_idx = (k + 1) * self.n_components
                    if X_transformed[i, start_idx:end_idx].abs().sum() > 0:
                        assignments[i] = k
                        break
                        
        # Reconstruct each point
        for k, rep in enumerate(self.representations):
            cluster_mask = (assignments == k)
            if cluster_mask.any():
                start_idx = k * self.n_components  
                end_idx = (k + 1) * self.n_components
                local_coords = X_transformed[cluster_mask, start_idx:end_idx]
                
                # Reconstruct: x = μ + W * z
                reconstructed[cluster_mask] = (
                    rep.mean.unsqueeze(0) + 
                    torch.matmul(local_coords, rep.W.t())
                )
                
        return reconstructed