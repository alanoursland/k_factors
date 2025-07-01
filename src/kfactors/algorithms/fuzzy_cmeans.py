"""
Fuzzy C-Means clustering algorithm.

Soft clustering algorithm where each point has fractional membership
in all clusters. The fuzziness is controlled by parameter m.
"""

from typing import Optional, List, Union, Literal
import torch
from torch import Tensor
import warnings

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import ClusterRepresentation, ClusteringObjective
from ..representations.centroid import CentroidRepresentation
from ..assignments.fuzzy import FuzzyAssignment, PossibilisticAssignment, FuzzyPossibilisticAssignment
from ..updates.mean import MeanUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective, ParameterChange, CombinedCriterion


class FuzzyCMeansObjective(ClusteringObjective):
    """Fuzzy C-Means objective function.
    
    Minimizes: J = Σ_i Σ_k u_ik^m * d_ik^2
    where u_ik is membership and d_ik is distance.
    """
    
    def __init__(self, m: float = 2.0):
        """
        Args:
            m: Fuzziness exponent
        """
        self.m = m
        
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor, **kwargs) -> Tensor:
        """Compute fuzzy objective value."""
        # assignments is (n, K) membership matrix
        n_clusters = len(representations)
        
        total = 0.0
        for k, rep in enumerate(representations):
            distances = rep.distance_to_point(points)
            # Weighted sum with membership^m
            weighted_distances = (assignments[:, k] ** self.m) * distances
            total += weighted_distances.sum()
            
        return total
        
    @property
    def minimize(self) -> bool:
        return True


class FuzzyCMeans(BaseClusteringAlgorithm):
    """Fuzzy C-Means (FCM) clustering algorithm.
    
    FCM allows each data point to belong to multiple clusters with
    different degrees of membership. The algorithm iteratively updates
    memberships and cluster centers to minimize the fuzzy objective.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    m : float, default=2.0
        Fuzziness exponent. Must be > 1.
        - m=1: Equivalent to hard K-means
        - m=2: Standard FCM (recommended)
        - m→∞: All points have equal membership in all clusters
    init : {'k-means++', 'random'} or array-like, default='k-means++'
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
        Final cluster centers
    u_ : Tensor of shape (n_samples, n_clusters)
        Final membership matrix
    n_iter_ : int
        Number of iterations run
    inertia_ : float
        Final objective value
        
    Examples
    --------
    >>> import torch
    >>> from kfactors.algorithms import FuzzyCMeans
    >>> 
    >>> X = torch.randn(100, 2)
    >>> fcm = FuzzyCMeans(n_clusters=3, m=2.0)
    >>> fcm.fit(X)
    >>> 
    >>> # Get fuzzy memberships
    >>> memberships = fcm.u_
    >>> 
    >>> # Get hard clustering
    >>> labels = fcm.predict(X)
    """
    
    def __init__(self,
                 n_clusters: int,
                 m: float = 2.0,
                 init: Union[str, Tensor] = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize Fuzzy C-Means."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        if m <= 1:
            raise ValueError(f"Fuzziness parameter m must be > 1, got {m}")
            
        self.m = m
        self.init = init
        
        # FCM specific attributes
        self.u_ = None  # Membership matrix
        
    def _create_components(self) -> None:
        """Create FCM specific components."""
        # Fuzzy assignment
        self.assignment_strategy = FuzzyAssignment(m=self.m)
        
        # Weighted mean updater
        self.update_strategy = MeanUpdater()
        
        # Initialization
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            else:
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
        else:
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        self.convergence_criterion = ChangeInObjective(rel_tol=self.tol)
        
        # Combined objective
        class FPCMObjective(ClusteringObjective):
            def __init__(self, a: float, b: float, m_fuzzy: float, m_poss: float):
                self.a = a
                self.b = b
                self.m_fuzzy = m_fuzzy
                self.m_poss = m_poss
                
            def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                       assignments: Tensor, **kwargs) -> Tensor:
                """FPCM objective combines fuzzy and possibilistic terms."""
                aux_info = kwargs.get('aux_info', {})
                fuzzy_u = aux_info.get('fuzzy_memberships', assignments)
                typicality_t = aux_info.get('typicalities', assignments)
                eta = aux_info.get('eta', torch.ones(len(representations)))
                
                total = 0.0
                
                for k, rep in enumerate(representations):
                    distances = rep.distance_to_point(points)
                    
                    # Fuzzy term
                    fuzzy_term = torch.sum((fuzzy_u[:, k] ** self.m_fuzzy) * distances)
                    
                    # Possibilistic term
                    poss_term = torch.sum((typicality_t[:, k] ** self.m_poss) * distances)
                    poss_penalty = eta[k] * torch.sum((1 - typicality_t[:, k]) ** self.m_poss)
                    
                    # Combined
                    total += self.a * fuzzy_term + self.b * (poss_term + poss_penalty)
                    
                return total
                
            @property
            def minimize(self) -> bool:
                return True
                
        self.objective = FPCMObjective(self.a, self.b, self.m_fuzzy, self.m_poss)
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'FuzzyPossibilisticCMeans':
        """Fit FPCM clustering."""
        if self.verbose:
            print(f"Fuzzy-Possibilistic C-Means: {self.n_clusters} clusters")
            print(f"  Fuzzy m={self.m_fuzzy}, Possibilistic m={self.m_poss}")
            print(f"  Weights: a={self.a}, b={self.b}")
            
        super().fit(X, y)
        
        # Extract both membership types
        if self.history_:
            final_state = self.history_[-1]
            aux_info = final_state.metadata.get('aux_info', {})
            
            self.u_ = final_state.assignments.get_soft()  # Combined
            self.fuzzy_u_ = aux_info.get('fuzzy_memberships', self.u_)
            self.typicality_t_ = aux_info.get('typicalities', self.u_)
            self.eta_ = aux_info.get('eta', None)
            
        return self
        
    def get_outlier_scores(self) -> Tensor:
        """Get outlier scores based on typicalities.
        
        Returns
        -------
        scores : Tensor of shape (n_samples,)
            Outlier scores in [0, 1], higher means more outlier-like
        """
        if self.typicality_t_ is None:
            raise RuntimeError("Model must be fitted first")
            
        # Outlier score is 1 - max_typicality
        return 1.0 - self.typicality_t_.max(dim=1)[0] 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            elif self.init == 'random':
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
            else:
                raise ValueError(f"Unknown init method: {self.init}")
        else:
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        # Combined convergence
        self.convergence_criterion = CombinedCriterion([
            ChangeInObjective(rel_tol=self.tol),
            ParameterChange(tol=self.tol, parameter='mean')
        ], mode='any')
        
        # Objective
        self.objective = FuzzyCMeansObjective(m=self.m)
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create centroid representations."""
        representations = []
        dimension = data.shape[1]
        
        # Get initial centers
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        for init_rep in init_representations:
            centroid = CentroidRepresentation(dimension, self.device)
            centroid.mean = init_rep.get_parameters()['mean']
            representations.append(centroid)
            
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'FuzzyCMeans':
        """Fit Fuzzy C-Means clustering.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : FuzzyCMeans
            Fitted estimator
        """
        if self.verbose:
            print(f"Fuzzy C-Means: {self.n_clusters} clusters with m={self.m}")
            
        # Standard fit
        super().fit(X, y)
        
        # Extract final membership matrix
        if self.history_:
            final_state = self.history_[-1]
            self.u_ = final_state.assignments.get_soft()
            
        return self
        
    def predict(self, X: Tensor) -> Tensor:
        """Predict hard cluster labels.
        
        Assigns each point to the cluster with highest membership.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : Tensor of shape (n_samples,)
            Cluster labels
        """
        memberships = self.predict_proba(X)
        return torch.argmax(memberships, dim=1)
        
    def predict_proba(self, X: Tensor) -> Tensor:
        """Predict fuzzy memberships.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        memberships : Tensor of shape (n_samples, n_clusters)
            Fuzzy membership matrix
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        X = self._validate_data(X)
        
        memberships, _ = self.assignment_strategy.compute_assignments(
            X, self.representations
        )
        
        return memberships
        
    def transform(self, X: Tensor) -> Tensor:
        """Transform data to membership space.
        
        Returns the fuzzy membership values, which can be interpreted
        as a soft encoding of the data.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        memberships : Tensor of shape (n_samples, n_clusters)
            Fuzzy memberships
        """
        return self.predict_proba(X)
        
    def partition_coefficient(self) -> float:
        """Compute fuzzy partition coefficient.
        
        FPC measures the amount of overlap between clusters.
        Values close to 1 indicate well-separated clusters.
        
        Returns
        -------
        fpc : float
            Fuzzy partition coefficient in [1/K, 1]
        """
        if self.u_ is None:
            raise RuntimeError("Model must be fitted first")
            
        fpc = torch.sum(self.u_ ** 2) / len(self.u_)
        return fpc.item()
        
    def partition_entropy(self) -> float:
        """Compute fuzzy partition entropy.
        
        FPE measures the fuzziness of the clustering.
        Lower values indicate less fuzzy (more crisp) clustering.
        
        Returns
        -------
        fpe : float
            Fuzzy partition entropy
        """
        if self.u_ is None:
            raise RuntimeError("Model must be fitted first")
            
        # Avoid log(0)
        u_safe = self.u_.clamp(min=1e-10)
        fpe = -torch.sum(self.u_ * torch.log(u_safe)) / len(self.u_)
        return fpe.item()


class PossibilisticCMeans(FuzzyCMeans):
    """Possibilistic C-Means (PCM) clustering.
    
    Unlike FCM, memberships don't need to sum to 1, allowing
    for better outlier handling.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    m : float, default=2.0
        Fuzziness exponent
    eta : array-like, optional
        Typicality parameters. If None, estimated from data.
    eta_factor : float, default=1.0
        Multiplicative factor for automatic eta estimation
    Other parameters are same as FuzzyCMeans
    
    Attributes
    ----------
    eta_ : Tensor of shape (n_clusters,)
        Typicality parameters
    t_ : Tensor of shape (n_samples, n_clusters)
        Final typicality matrix
    """
    
    def __init__(self,
                 n_clusters: int,
                 m: float = 2.0,
                 eta: Optional[Tensor] = None,
                 eta_factor: float = 1.0,
                 init: Union[str, Tensor] = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize PCM."""
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            init=init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.eta = eta
        self.eta_factor = eta_factor
        self.eta_ = None
        self.t_ = None
        
    def _create_components(self) -> None:
        """Create PCM specific components."""
        # Possibilistic assignment
        self.assignment_strategy = PossibilisticAssignment(
            m=self.m,
            eta=self.eta,
            eta_factor=self.eta_factor
        )
        
        # Rest is same as FCM
        self.update_strategy = MeanUpdater()
        
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            else:
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
        else:
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        self.convergence_criterion = ChangeInObjective(rel_tol=self.tol)
        
        # Modified objective for PCM
        class PCMObjective(ClusteringObjective):
            def __init__(self, m: float):
                self.m = m
                
            def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                       assignments: Tensor, **kwargs) -> Tensor:
                """PCM objective includes typicality penalty."""
                total = 0.0
                eta = kwargs.get('aux_info', {}).get('eta', torch.ones(len(representations)))
                
                for k, rep in enumerate(representations):
                    distances = rep.distance_to_point(points)
                    # Typicality term
                    t_k = assignments[:, k]
                    total += torch.sum((t_k ** self.m) * distances)
                    # Penalty term
                    total += eta[k] * torch.sum((1 - t_k) ** self.m)
                    
                return total
                
            @property
            def minimize(self) -> bool:
                return True
                
        self.objective = PCMObjective(m=self.m)
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'PossibilisticCMeans':
        """Fit PCM clustering."""
        if self.verbose:
            print(f"Possibilistic C-Means: {self.n_clusters} clusters with m={self.m}")
            
        super().fit(X, y)
        
        # Extract typicality matrix and eta
        if self.history_:
            final_state = self.history_[-1]
            self.t_ = final_state.assignments.get_soft()
            self.u_ = self.t_  # For compatibility
            
            # Get eta from aux_info
            aux_info = final_state.metadata.get('aux_info', {})
            self.eta_ = aux_info.get('eta', None)
            
        return self
        
    def detect_outliers(self, threshold: float = 0.5) -> Tensor:
        """Detect outliers based on typicality.
        
        Points with max typicality below threshold are outliers.
        
        Parameters
        ----------
        threshold : float, default=0.5
            Typicality threshold
            
        Returns
        -------
        outliers : Tensor of shape (n_samples,)
            Boolean mask of outliers
        """
        if self.t_ is None:
            raise RuntimeError("Model must be fitted first")
            
        max_typicality = self.t_.max(dim=1)[0]
        return max_typicality < threshold


class FuzzyPossibilisticCMeans(FuzzyCMeans):
    """Fuzzy-Possibilistic C-Means (FPCM) clustering.
    
    Combines fuzzy memberships with possibilistic typicalities
    for robust clustering with outlier handling.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    m_fuzzy : float, default=2.0
        Fuzziness for memberships
    m_poss : float, default=2.0
        Fuzziness for typicalities
    a : float, default=1.0
        Weight for fuzzy term
    b : float, default=1.0
        Weight for possibilistic term
    Other parameters same as FuzzyCMeans
    """
    
    def __init__(self,
                 n_clusters: int,
                 m_fuzzy: float = 2.0,
                 m_poss: float = 2.0,
                 a: float = 1.0,
                 b: float = 1.0,
                 eta: Optional[Tensor] = None,
                 init: Union[str, Tensor] = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize FPCM."""
        # Use fuzzy m as primary
        super().__init__(
            n_clusters=n_clusters,
            m=m_fuzzy,
            init=init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.m_fuzzy = m_fuzzy
        self.m_poss = m_poss
        self.a = a
        self.b = b
        self.eta = eta
        
        # Additional attributes
        self.fuzzy_u_ = None
        self.typicality_t_ = None
        
    def _create_components(self) -> None:
        """Create FPCM components."""
        # Combined assignment
        self.assignment_strategy = FuzzyPossibilisticAssignment(
            m_fuzzy=self.m_fuzzy,
            m_poss=self.m_poss,
            a=self.a,
            b=self.b,
            eta=self.eta
        )
        
        # Rest similar to FCM
        self.update_strategy = MeanUpdater()
        
        if isinstance(self.init, str):
            if self.init ==