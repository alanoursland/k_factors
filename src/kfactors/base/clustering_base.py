"""
Base class for clustering algorithms in the K-Factors family.

Provides the common algorithmic skeleton for alternating optimization
between assignment and update steps.
"""

from abc import abstractmethod
from typing import Optional, Dict, Any, List, Callable, Union
import torch
from torch import Tensor
import time
import warnings

from .interfaces import (
    ClusterRepresentation, AssignmentStrategy, ParameterUpdater,
    InitializationStrategy, ConvergenceCriterion, ClusteringObjective
)
from .data_structures import (
    ClusterState, AssignmentMatrix, AlgorithmState, DirectionTracker
)


class BaseClusteringAlgorithm:
    """Base class implementing the alternating optimization framework.
    
    Subclasses need to specify:
    - Cluster representation type
    - Assignment strategy  
    - Parameter update strategy
    - Initialization strategy
    - Convergence criterion
    - Objective function
    """
    
    def __init__(self,
                 n_clusters: int,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            n_clusters: Number of clusters K
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            random_state: Random seed for reproducibility
            device: Torch device (None for auto-detect)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                
        # These will be set by subclasses
        self.representations: Optional[List[ClusterRepresentation]] = None
        self.assignment_strategy: Optional[AssignmentStrategy] = None
        self.update_strategy: Optional[ParameterUpdater] = None
        self.initialization_strategy: Optional[InitializationStrategy] = None
        self.convergence_criterion: Optional[ConvergenceCriterion] = None
        self.objective: Optional[ClusteringObjective] = None
        
        # Algorithm state
        self.fitted_ = False
        self.n_iter_ = 0
        self.history_ = []
        
    @abstractmethod
    def _create_components(self) -> None:
        """Create algorithm-specific components.
        
        Subclasses must implement this to instantiate:
        - self.assignment_strategy
        - self.update_strategy  
        - self.initialization_strategy
        - self.convergence_criterion
        - self.objective
        """
        pass
        
    @abstractmethod
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create cluster representations after initialization.
        
        Args:
            data: (n, d) data tensor
            
        Returns:
            List of K cluster representations
        """
        pass
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'BaseClusteringAlgorithm':
        """Fit the clustering model.
        
        Args:
            X: (n, d) data tensor
            y: Ignored (for sklearn compatibility)
            
        Returns:
            Self
        """
        return self._fit(X)
        
    def fit_predict(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Fit and return cluster assignments.
        
        Args:
            X: (n, d) data tensor
            y: Ignored
            
        Returns:
            (n,) tensor of cluster assignments
        """
        self._fit(X)
        return self.predict(X)
        
    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster assignments for new data.
        
        Args:
            X: (n, d) data tensor
            
        Returns:
            (n,) tensor of cluster assignments
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict")
            
        X = self._validate_data(X)
        
        # Compute assignments using current representations
        assignments = self.assignment_strategy.compute_assignments(
            X, self.representations
        )
        
        if isinstance(assignments, tuple):
            assignments = assignments[0]
            
        assignment_matrix = AssignmentMatrix(
            assignments, 
            self.n_clusters,
            is_soft=self.assignment_strategy.is_soft
        )
        
        return assignment_matrix.get_hard()
        
    def _fit(self, X: Tensor) -> 'BaseClusteringAlgorithm':
        """Internal fit method implementing the alternating optimization."""
        # Validate and prepare data
        X = self._validate_data(X)
        n_points, dimension = X.shape
        
        # Create algorithm components
        self._create_components()
        
        # Initialize clusters
        if self.verbose:
            print(f"Initializing {self.n_clusters} clusters...")
            
        start_time = time.time()
        self.representations = self.initialization_strategy.initialize(
            X, self.n_clusters
        )
        
        # Create any algorithm-specific representations
        self.representations = self._create_representations(X)
        
        # Initialize state tracking
        self.n_iter_ = 0
        self.history_ = []
        self.convergence_criterion.reset()
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            iter_start_time = time.time()
            
            # Assignment step
            assignment_result = self.assignment_strategy.compute_assignments(
                X, self.representations
            )
            
            if isinstance(assignment_result, tuple):
                assignments, aux_info = assignment_result
            else:
                assignments = assignment_result
                aux_info = {}
                
            assignment_matrix = AssignmentMatrix(
                assignments,
                self.n_clusters, 
                is_soft=self.assignment_strategy.is_soft
            )
            
            # Update step
            for k, representation in enumerate(self.representations):
                if self.assignment_strategy.is_soft:
                    # Get all points with their weights
                    cluster_weights = assignment_matrix.get_cluster_weights(k)
                    mask = cluster_weights > 1e-10  # Numerical threshold
                    if mask.sum() > 0:
                        self.update_strategy.update(
                            representation,
                            X[mask],
                            cluster_weights[mask],
                            **aux_info
                        )
                else:
                    # Get points assigned to this cluster
                    cluster_indices = assignment_matrix.get_cluster_indices(k)
                    if len(cluster_indices) > 0:
                        cluster_points = X[cluster_indices]
                        self.update_strategy.update(
                            representation,
                            cluster_points,
                            None,  # No weights for hard assignment
                            **aux_info
                        )
                        
            # Compute objective
            objective_value = self.objective.compute(
                X, self.representations, assignments
            )
            
            # Track state
            cluster_state = self._extract_cluster_state()
            algorithm_state = AlgorithmState(
                iteration=iteration,
                cluster_state=cluster_state,
                assignments=assignment_matrix,
                objective_value=objective_value.item(),
                metadata={'aux_info': aux_info}
            )
            self.history_.append(algorithm_state)
            
            # Check convergence
            converged = self.convergence_criterion.check({
                'iteration': iteration,
                'objective': objective_value.item(),
                'assignments': assignments,
                'cluster_state': cluster_state
            })
            
            # Logging
            iter_time = time.time() - iter_start_time
            if self.verbose >= 2 or (self.verbose >= 1 and iteration % 10 == 0):
                obj_direction = "↓" if self.objective.minimize else "↑"
                print(f"Iteration {iteration:3d}: objective = {objective_value.item():.6f} "
                      f"{obj_direction} ({iter_time:.3f}s)")
                      
            if converged:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            self.n_iter_ = iteration + 1
            
        total_time = time.time() - start_time
        
        if self.verbose:
            if not converged:
                warnings.warn(f"Failed to converge after {self.max_iter} iterations")
            print(f"Total fitting time: {total_time:.3f}s")
            
        self.fitted_ = True
        return self
        
    def _validate_data(self, X: Tensor) -> Tensor:
        """Validate and prepare input data."""
        if not isinstance(X, Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {X.dim()}D")
            
        # Move to correct device
        X = X.to(self.device)
        
        # Ensure float32
        if X.dtype != torch.float32:
            X = X.float()
            
        return X
        
    def _extract_cluster_state(self) -> ClusterState:
        """Extract current cluster parameters into ClusterState object."""
        # Collect means
        means = torch.stack([
            rep.get_parameters()['mean'] 
            for rep in self.representations
        ])
        
        state = ClusterState(
            means=means,
            n_clusters=self.n_clusters,
            dimension=means.shape[1]
        )
        
        # Add other parameters if present
        first_rep_params = self.representations[0].get_parameters()
        
        if 'basis' in first_rep_params:
            bases = []
            for rep in self.representations:
                basis = rep.get_parameters()['basis']
                bases.append(basis)
            state.bases = torch.stack(bases)
            
        if 'covariance' in first_rep_params:
            covs = torch.stack([
                rep.get_parameters()['covariance']
                for rep in self.representations
            ])
            state.covariances = covs
            
        if 'variance' in first_rep_params:
            vars = torch.stack([
                rep.get_parameters()['variance']
                for rep in self.representations  
            ])
            state.variances = vars
            
        return state
        
    @property
    def cluster_centers_(self) -> Tensor:
        """Get cluster centers/means."""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
        return self._extract_cluster_state().means
        
    @property 
    def inertia_(self) -> float:
        """Get final objective value."""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
        return self.history_[-1].objective_value
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters (sklearn compatibility)."""
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'device': self.device
        }
        
    def set_params(self, **params) -> 'BaseClusteringAlgorithm':
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self