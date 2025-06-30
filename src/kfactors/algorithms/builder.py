"""
Builder pattern for constructing clustering algorithms.

Provides a fluent interface for creating custom clustering algorithms
by combining different components.
"""

from typing import Optional, Union, Type, Any
import torch
from torch import Tensor

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import (
    ClusterRepresentation, AssignmentStrategy, ParameterUpdater,
    InitializationStrategy, ConvergenceCriterion, ClusteringObjective
)

# Import all components
from ..representations import (
    CentroidRepresentation, SubspaceRepresentation, PPCARepresentation
)
from ..assignments import HardAssignment, PenalizedAssignment
from ..updates import MeanUpdater, PCAUpdater, SequentialPCAUpdater
from ..initialization import RandomInit, KMeansPlusPlusInit
from ..utils.convergence import (
    ChangeInAssignments, ChangeInObjective, ParameterChange, CombinedCriterion
)


class CustomClusteringAlgorithm(BaseClusteringAlgorithm):
    """Custom clustering algorithm built from components."""
    
    def __init__(self,
                 n_clusters: int,
                 representation_class: Type[ClusterRepresentation],
                 representation_kwargs: dict,
                 assignment_strategy: AssignmentStrategy,
                 update_strategy: ParameterUpdater,
                 initialization_strategy: InitializationStrategy,
                 convergence_criterion: ConvergenceCriterion,
                 objective: ClusteringObjective,
                 max_iter: int = 100,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize custom algorithm with provided components."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=1e-4,  # Not used directly, criterion handles convergence
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.representation_class = representation_class
        self.representation_kwargs = representation_kwargs
        self.assignment_strategy = assignment_strategy
        self.update_strategy = update_strategy
        self.initialization_strategy = initialization_strategy
        self.convergence_criterion = convergence_criterion
        self.objective = objective
        
    def _create_components(self) -> None:
        """Components are already provided."""
        pass
        
    def _create_representations(self, data: Tensor) -> list[ClusterRepresentation]:
        """Create representations using provided class."""
        dimension = data.shape[1]
        
        # Get initial centers
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        representations = []
        for init_rep in init_representations:
            # Create new representation
            rep = self.representation_class(
                dimension=dimension,
                device=self.device,
                **self.representation_kwargs
            )
            
            # Copy initial mean
            rep.mean = init_rep.get_parameters()['mean']
            
            representations.append(rep)
            
        return representations


class ClusteringBuilder:
    """Fluent builder for creating clustering algorithms.
    
    Examples
    --------
    >>> # Build a custom K-means variant
    >>> algorithm = (ClusteringBuilder()
    ...     .with_representation(CentroidRepresentation)
    ...     .with_hard_assignment()
    ...     .with_mean_update()
    ...     .with_kmeans_plusplus_init()
    ...     .with_assignment_convergence(tol=0.001)
    ...     .build(n_clusters=5))
    
    >>> # Build a subspace clustering algorithm
    >>> algorithm = (ClusteringBuilder()
    ...     .with_subspace_representation(dim=2)
    ...     .with_hard_assignment()
    ...     .with_pca_update()
    ...     .with_random_init()
    ...     .with_objective_convergence(tol=1e-4)
    ...     .build(n_clusters=3))
    """
    
    def __init__(self):
        """Initialize builder with defaults."""
        # Defaults
        self._representation_class = CentroidRepresentation
        self._representation_kwargs = {}
        self._assignment_strategy = HardAssignment()
        self._update_strategy = MeanUpdater()
        self._initialization_strategy = KMeansPlusPlusInit()
        self._convergence_criterion = ChangeInObjective(rel_tol=1e-4)
        self._objective = None  # Will be set based on representation
        
        # Algorithm parameters
        self._max_iter = 100
        self._verbose = 0
        self._random_state = None
        self._device = None
        
    def with_representation(self, representation_class: Type[ClusterRepresentation],
                          **kwargs) -> 'ClusteringBuilder':
        """Set the cluster representation type."""
        self._representation_class = representation_class
        self._representation_kwargs = kwargs
        return self
        
    def with_centroid_representation(self) -> 'ClusteringBuilder':
        """Use centroid representation (K-means style)."""
        return self.with_representation(CentroidRepresentation)
        
    def with_subspace_representation(self, dim: int) -> 'ClusteringBuilder':
        """Use subspace representation."""
        return self.with_representation(SubspaceRepresentation, subspace_dim=dim)
        
    def with_ppca_representation(self, latent_dim: int) -> 'ClusteringBuilder':
        """Use PPCA representation."""
        return self.with_representation(PPCARepresentation, latent_dim=latent_dim)
        
    def with_assignment_strategy(self, strategy: AssignmentStrategy) -> 'ClusteringBuilder':
        """Set the assignment strategy."""
        self._assignment_strategy = strategy
        return self
        
    def with_hard_assignment(self) -> 'ClusteringBuilder':
        """Use hard assignment."""
        return self.with_assignment_strategy(HardAssignment())
        
    def with_penalized_assignment(self, penalty_type: str = 'product',
                                 penalty_weight: float = 1.0) -> 'ClusteringBuilder':
        """Use penalized assignment (K-Factors style)."""
        return self.with_assignment_strategy(
            PenalizedAssignment(penalty_type, penalty_weight)
        )
        
    def with_update_strategy(self, strategy: ParameterUpdater) -> 'ClusteringBuilder':
        """Set the parameter update strategy."""
        self._update_strategy = strategy
        return self
        
    def with_mean_update(self) -> 'ClusteringBuilder':
        """Use simple mean update."""
        return self.with_update_strategy(MeanUpdater())
        
    def with_pca_update(self, n_components: Optional[int] = None) -> 'ClusteringBuilder':
        """Use PCA update."""
        return self.with_update_strategy(PCAUpdater(n_components))
        
    def with_sequential_pca_update(self) -> 'ClusteringBuilder':
        """Use sequential PCA update (K-Factors style)."""
        return self.with_update_strategy(SequentialPCAUpdater())
        
    def with_initialization(self, strategy: InitializationStrategy) -> 'ClusteringBuilder':
        """Set initialization strategy."""
        self._initialization_strategy = strategy
        return self
        
    def with_random_init(self) -> 'ClusteringBuilder':
        """Use random initialization."""
        return self.with_initialization(RandomInit())
        
    def with_kmeans_plusplus_init(self) -> 'ClusteringBuilder':
        """Use K-means++ initialization."""
        return self.with_initialization(KMeansPlusPlusInit())
        
    def with_convergence_criterion(self, criterion: ConvergenceCriterion) -> 'ClusteringBuilder':
        """Set convergence criterion."""
        self._convergence_criterion = criterion
        return self
        
    def with_assignment_convergence(self, tol: float = 1e-4,
                                   patience: int = 1) -> 'ClusteringBuilder':
        """Use convergence based on assignment changes."""
        return self.with_convergence_criterion(
            ChangeInAssignments(min_change_fraction=tol, patience=patience)
        )
        
    def with_objective_convergence(self, rel_tol: float = 1e-4, 
                                  abs_tol: float = 1e-8,
                                  patience: int = 1) -> 'ClusteringBuilder':
        """Use convergence based on objective changes."""
        return self.with_convergence_criterion(
            ChangeInObjective(rel_tol=rel_tol, abs_tol=abs_tol, patience=patience)
        )
        
    def with_parameter_convergence(self, tol: float = 1e-6,
                                  parameter: str = 'mean',
                                  patience: int = 1) -> 'ClusteringBuilder':
        """Use convergence based on parameter changes."""
        return self.with_convergence_criterion(
            ParameterChange(tol=tol, parameter=parameter, patience=patience)
        )
        
    def with_combined_convergence(self, criteria: list[ConvergenceCriterion],
                                 mode: str = 'any') -> 'ClusteringBuilder':
        """Use combined convergence criteria."""
        return self.with_convergence_criterion(
            CombinedCriterion(criteria, mode=mode)
        )
        
    def with_objective(self, objective: ClusteringObjective) -> 'ClusteringBuilder':
        """Set the objective function."""
        self._objective = objective
        return self
        
    def with_max_iter(self, max_iter: int) -> 'ClusteringBuilder':
        """Set maximum iterations."""
        self._max_iter = max_iter
        return self
        
    def with_verbose(self, verbose: int) -> 'ClusteringBuilder':
        """Set verbosity level."""
        self._verbose = verbose
        return self
        
    def with_random_state(self, random_state: int) -> 'ClusteringBuilder':
        """Set random seed."""
        self._random_state = random_state
        return self
        
    def with_device(self, device: Union[str, torch.device]) -> 'ClusteringBuilder':
        """Set computation device."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self
        
    def build(self, n_clusters: int) -> CustomClusteringAlgorithm:
        """Build the clustering algorithm.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        algorithm : CustomClusteringAlgorithm
            The configured clustering algorithm
        """
        # Auto-configure objective if not set
        if self._objective is None:
            self._objective = self._create_default_objective()
            
        return CustomClusteringAlgorithm(
            n_clusters=n_clusters,
            representation_class=self._representation_class,
            representation_kwargs=self._representation_kwargs,
            assignment_strategy=self._assignment_strategy,
            update_strategy=self._update_strategy,
            initialization_strategy=self._initialization_strategy,
            convergence_criterion=self._convergence_criterion,
            objective=self._objective,
            max_iter=self._max_iter,
            verbose=self._verbose,
            random_state=self._random_state,
            device=self._device
        )
        
    def _create_default_objective(self) -> ClusteringObjective:
        """Create default objective based on representation type."""
        from ..distances import EuclideanDistance, OrthogonalDistance
        
        class DefaultObjective(ClusteringObjective):
            def __init__(self, representation_class):
                self.representation_class = representation_class
                
                # Choose distance based on representation
                if representation_class == CentroidRepresentation:
                    self.distance = EuclideanDistance()
                elif representation_class == SubspaceRepresentation:
                    self.distance = OrthogonalDistance()
                elif representation_class == PPCARepresentation:
                    from ..distances import LowRankMahalanobisDistance
                    self.distance = LowRankMahalanobisDistance()
                else:
                    # Fallback to Euclidean
                    self.distance = EuclideanDistance()
                    
            def compute(self, points: Tensor, representations: list[ClusterRepresentation],
                       assignments: Tensor, **kwargs) -> Tensor:
                total = 0.0
                
                for k, rep in enumerate(representations):
                    cluster_mask = (assignments == k)
                    if cluster_mask.any():
                        cluster_points = points[cluster_mask]
                        distances = self.distance.compute(cluster_points, rep)
                        total += distances.sum()
                        
                return total
                
            @property
            def minimize(self) -> bool:
                return True
                
        return DefaultObjective(self._representation_class)


def create_kmeans(n_clusters: int, **kwargs) -> CustomClusteringAlgorithm:
    """Create a K-means algorithm using the builder.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    **kwargs : dict
        Additional parameters passed to builder
        
    Returns
    -------
    algorithm : CustomClusteringAlgorithm
        K-means algorithm
    """
    builder = ClusteringBuilder()
    builder.with_centroid_representation()
    builder.with_hard_assignment()
    builder.with_mean_update()
    builder.with_kmeans_plusplus_init()
    builder.with_assignment_convergence()
    
    # Apply any custom parameters
    for key, value in kwargs.items():
        method = getattr(builder, f'with_{key}', None)
        if method:
            method(value)
            
    return builder.build(n_clusters)


def create_ksubspaces(n_clusters: int, subspace_dim: int, **kwargs) -> CustomClusteringAlgorithm:
    """Create a K-Subspaces algorithm using the builder.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    subspace_dim : int
        Dimension of each subspace
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    algorithm : CustomClusteringAlgorithm
        K-Subspaces algorithm
    """
    builder = ClusteringBuilder()
    builder.with_subspace_representation(dim=subspace_dim)
    builder.with_hard_assignment()
    builder.with_pca_update(n_components=subspace_dim)
    builder.with_kmeans_plusplus_init()
    builder.with_objective_convergence()
    
    for key, value in kwargs.items():
        method = getattr(builder, f'with_{key}', None)
        if method:
            method(value)
            
    return builder.build(n_clusters)


def create_kfactors_variant(n_clusters: int, n_components: int,
                           penalty_weight: float = 0.9, **kwargs) -> CustomClusteringAlgorithm:
    """Create a K-Factors variant using the builder.
    
    Note: This creates a simplified variant. For full K-Factors with
    sequential extraction, use the KFactors class directly.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    n_components : int
        Number of components per cluster
    penalty_weight : float
        Weight for penalty term
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    algorithm : CustomClusteringAlgorithm
        K-Factors-like algorithm
    """
    builder = ClusteringBuilder()
    builder.with_ppca_representation(latent_dim=n_components)
    builder.with_penalized_assignment(penalty_weight=penalty_weight)
    builder.with_pca_update(n_components=n_components)
    builder.with_kmeans_plusplus_init()
    builder.with_objective_convergence()
    
    for key, value in kwargs.items():
        method = getattr(builder, f'with_{key}', None)
        if method:
            method(value)
            
    return builder.build(n_clusters)