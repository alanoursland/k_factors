"""
Gaussian Mixture Model (GMM) clustering.

Probabilistic clustering using a mixture of multivariate Gaussians
with various covariance structures.
"""

from typing import Optional, List, Union, Literal, Dict, Tuple
import torch
from torch import Tensor
import warnings

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import ClusterRepresentation, ClusteringObjective
from ..representations.gaussian import GaussianRepresentation
from ..assignments.soft import GaussianSoftAssignment
from ..updates.em_updates import GaussianEMUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective


class GMMObjective(ClusteringObjective):
    """GMM objective: negative log-likelihood."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor, **kwargs) -> Tensor:
        """Compute negative log-likelihood.
        
        Note: We return negative because the framework minimizes objectives.
        """
        aux_info = kwargs.get('aux_info', {})
        
        # Use pre-computed log-likelihood from E-step if available
        if 'total_log_likelihood' in aux_info:
            return -aux_info['total_log_likelihood']
            
        # Otherwise compute it
        log_marginal = aux_info.get('log_marginal', None)
        if log_marginal is not None:
            return -log_marginal.sum()
            
        # Fallback: compute from scratch
        n_clusters = len(representations)
        mixing_weights = aux_info.get('mixing_weights', 
                                    torch.ones(n_clusters) / n_clusters)
        
        log_likelihood = 0.0
        for i, point in enumerate(points):
            point_ll = torch.logsumexp(
                torch.stack([
                    rep.log_likelihood(point.unsqueeze(0)).squeeze() + torch.log(mixing_weights[k])
                    for k, rep in enumerate(representations)
                ]), dim=0
            )
            log_likelihood += point_ll
            
        return -log_likelihood
        
    @property
    def minimize(self) -> bool:
        return True  # Minimize negative log-likelihood


class GaussianMixture(BaseClusteringAlgorithm):
    """Gaussian Mixture Model clustering.
    
    Fits a mixture of Gaussians using the EM algorithm. Supports various
    covariance structures and initialization methods.
    
    Parameters
    ----------
    n_clusters : int
        Number of Gaussian components
    covariance_type : {'full', 'diagonal', 'spherical', 'tied'}, default='full'
        Type of covariance parameters:
        - 'full': Each component has its own general covariance matrix
        - 'diagonal': Each component has its own diagonal covariance
        - 'spherical': Each component has single variance (σ²I)
        - 'tied': All components share the same covariance matrix
    init : {'k-means++', 'random', 'kmeans'} or array-like, default='k-means++'
        Initialization method:
        - 'k-means++': Use K-means++ for initial centers
        - 'random': Random selection of data points
        - 'kmeans': Run K-means first
        - array of shape (n_clusters, n_features): Use as initial means
    n_init : int, default=1
        Number of initializations to try
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence tolerance on log-likelihood
    reg_covar : float, default=1e-6
        Regularization added to diagonal of covariance matrices
    warm_start : bool, default=False
        If True, use previous solution as initialization
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed
    device : torch.device, optional
        Computation device
        
    Attributes
    ----------
    means_ : Tensor of shape (n_clusters, n_features)
        Mean of each mixture component
    covariances_ : Tensor
        Covariance of each mixture component. Shape depends on
        covariance_type:
        - 'full': (n_clusters, n_features, n_features)
        - 'diagonal': (n_clusters, n_features)
        - 'spherical': (n_clusters,)
        - 'tied': (n_features, n_features)
    weights_ : Tensor of shape (n_clusters,)
        Mixing weights of each component
    converged_ : bool
        True if EM converged
    n_iter_ : int
        Number of EM iterations run
    lower_bound_ : float
        Log-likelihood lower bound (ELBO) value
        
    Examples
    --------
    >>> import torch
    >>> from kfactors.algorithms import GaussianMixture
    >>> 
    >>> # Generate mixture data
    >>> X = torch.cat([
    ...     torch.randn(100, 2),
    ...     torch.randn(100, 2) + 3
    ... ])
    >>> 
    >>> # Fit GMM
    >>> gmm = GaussianMixture(n_clusters=2, covariance_type='full')
    >>> gmm.fit(X)
    >>> 
    >>> # Predict cluster probabilities
    >>> probs = gmm.predict_proba(X)
    >>> 
    >>> # Generate new samples
    >>> X_new, y_new = gmm.sample(100)
    """
    
    def __init__(self,
                 n_clusters: int,
                 covariance_type: Literal['full', 'diagonal', 'spherical', 'tied'] = 'full',
                 init: Union[str, Tensor] = 'k-means++',
                 n_init: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 reg_covar: float = 1e-6,
                 warm_start: bool = False,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize Gaussian Mixture Model."""
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.covariance_type = covariance_type
        self.init = init
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.warm_start = warm_start
        
        # GMM specific attributes
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.converged_ = False
        self.lower_bound_ = -float('inf')
        
        # For warm start
        self._prev_means = None
        self._prev_covariances = None
        self._prev_weights = None
        
    def _create_components(self) -> None:
        """Create GMM specific components."""
        # Soft assignment for E-step
        self.assignment_strategy = GaussianSoftAssignment()
        
        # EM updater for M-step
        self.update_strategy = GaussianEMUpdater(
            min_covar=self.reg_covar,
            covariance_type=self.covariance_type
        )
        
        # Initialization
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                self.initialization_strategy = KMeansPlusPlusInit()
            elif self.init == 'random':
                from ..initialization.random import RandomInit
                self.initialization_strategy = RandomInit()
            elif self.init == 'kmeans':
                # Will handle in fit()
                self.initialization_strategy = None
            else:
                raise ValueError(f"Unknown init method: {self.init}")
        else:
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        # Convergence based on log-likelihood
        self.convergence_criterion = ChangeInObjective(
            rel_tol=self.tol,
            abs_tol=1e-9,
            patience=1
        )
        
        # Objective
        self.objective = GMMObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create Gaussian representations."""
        representations = []
        dimension = data.shape[1]
        
        # Handle different initialization methods
        if self.warm_start and self._prev_means is not None:
            # Use previous solution
            for k in range(self.n_clusters):
                gaussian = GaussianRepresentation(
                    dimension=dimension,
                    device=self.device,
                    covariance_type=self.covariance_type,
                    regularization=self.reg_covar
                )
                gaussian.mean = self._prev_means[k]
                
                if self.covariance_type == 'full':
                    gaussian.covariance = self._prev_covariances[k]
                elif self.covariance_type == 'diagonal':
                    gaussian.variance = self._prev_covariances[k]
                elif self.covariance_type == 'spherical':
                    gaussian.variance = self._prev_covariances[k]
                    
                representations.append(gaussian)
                
        elif self.init == 'kmeans':
            # Run K-means for initialization
            from .kmeans import KMeans
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=10,
                random_state=self.random_state,
                device=self.device,
                verbose=0
            )
            kmeans.fit(data)
            
            # Create Gaussians from K-means results
            for k in range(self.n_clusters):
                gaussian = GaussianRepresentation(
                    dimension=dimension,
                    device=self.device,
                    covariance_type=self.covariance_type,
                    regularization=self.reg_covar
                )
                gaussian.mean = kmeans.cluster_centers_[k]
                
                # Initialize covariance from cluster
                cluster_mask = kmeans.labels_ == k
                if cluster_mask.any():
                    cluster_points = data[cluster_mask]
                    gaussian.update_from_points(cluster_points)
                    
                representations.append(gaussian)
                
        else:
            # Use standard initialization
            init_representations = self.initialization_strategy.initialize(
                data, self.n_clusters
            )
            
            for init_rep in init_representations:
                gaussian = GaussianRepresentation(
                    dimension=dimension,
                    device=self.device,
                    covariance_type=self.covariance_type,
                    regularization=self.reg_covar
                )
                gaussian.mean = init_rep.get_parameters()['mean']
                representations.append(gaussian)
                
        # Initialize mixing weights
        self.weights_ = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
        
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'GaussianMixture':
        """Fit Gaussian Mixture Model.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : GaussianMixture
            Fitted estimator
        """
        X = self._validate_data(X)
        
        # Multiple initialization runs
        best_log_likelihood = -float('inf')
        best_state = None
        
        for init_iter in range(self.n_init):
            if self.verbose and self.n_init > 1:
                print(f"\nInitialization {init_iter + 1}/{self.n_init}")
                
            # Reset for this initialization
            self.converged_ = False
            self.n_iter_ = 0
            
            # Run EM
            self._single_fit(X)
            
            # Check if this is the best so far
            if self.lower_bound_ > best_log_likelihood:
                best_log_likelihood = self.lower_bound_
                best_state = self._extract_parameters()
                
        # Use best initialization
        if best_state is not None:
            self._set_parameters(best_state)
            self.lower_bound_ = best_log_likelihood
            
        self.fitted_ = True
        return self
        
    def _single_fit(self, X: Tensor) -> None:
        """Single run of EM algorithm."""
        # Create components and representations
        self._create_components()
        self.representations = self._create_representations(X)
        
        # Main EM loop
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities, aux_info = self.assignment_strategy.compute_assignments(
                X, self.representations, mixing_weights=self.weights_
            )
            
            # M-step: update parameters
            # Update each Gaussian
            for k, gaussian in enumerate(self.representations):
                resp_k = responsibilities[:, k]
                
                # For tied covariance, need special handling
                if self.covariance_type == 'tied':
                    self.update_strategy.update(
                        gaussian, X, resp_k,
                        all_points=X,
                        all_assignments=responsibilities,
                        representations=self.representations
                    )
                else:
                    self.update_strategy.update(gaussian, X, resp_k)
                    
            # Update mixing weights
            self.weights_ = responsibilities.sum(dim=0) / len(X)
            self.weights_ = torch.clamp(self.weights_, min=1e-10)
            self.weights_ = self.weights_ / self.weights_.sum()
            
            # Compute objective (negative log-likelihood)
            objective = self.objective.compute(X, self.representations, 
                                             responsibilities, aux_info=aux_info)
            
            # Check convergence
            converged = self.convergence_criterion.check({
                'iteration': iteration,
                'objective': objective.item()
            })
            
            if self.verbose >= 2:
                print(f"  Iteration {iteration:3d}: log-likelihood = {-objective.item():.6f}")
                
            if converged:
                self.converged_ = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            self.n_iter_ = iteration + 1
            
        self.lower_bound_ = -objective.item()
        
        # Extract final parameters
        self._extract_parameters()
        
    def _extract_parameters(self) -> Dict[str, Tensor]:
        """Extract parameters from representations."""
        # Means
        self.means_ = torch.stack([rep.mean for rep in self.representations])
        
        # Covariances
        if self.covariance_type == 'full':
            self.covariances_ = torch.stack([
                rep.covariance for rep in self.representations
            ])
        elif self.covariance_type == 'diagonal':
            self.covariances_ = torch.stack([
                rep.variance for rep in self.representations
            ])
        elif self.covariance_type == 'spherical':
            self.covariances_ = torch.stack([
                rep.variance for rep in self.representations
            ])
        elif self.covariance_type == 'tied':
            # All share same covariance
            self.covariances_ = self.representations[0].covariance
            
        # Store for warm start
        if self.warm_start:
            self._prev_means = self.means_.clone()
            self._prev_covariances = self.covariances_.clone()
            self._prev_weights = self.weights_.clone()
            
        return {
            'means': self.means_,
            'covariances': self.covariances_,
            'weights': self.weights_
        }
        
    def _set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set parameters from dictionary."""
        self.means_ = params['means']
        self.covariances_ = params['covariances']
        self.weights_ = params['weights']
        
        # Update representations
        for k, rep in enumerate(self.representations):
            rep.mean = self.means_[k]
            if self.covariance_type == 'full':
                rep.covariance = self.covariances_[k]
            elif self.covariance_type == 'diagonal':
                rep.variance = self.covariances_[k]
            elif self.covariance_type == 'spherical':
                rep.variance = self.covariances_[k]
            elif self.covariance_type == 'tied':
                rep.covariance = self.covariances_
                
    def predict_proba(self, X: Tensor) -> Tensor:
        """Predict posterior probability of each component.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        probs : Tensor of shape (n_samples, n_clusters)
            Posterior probabilities
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        X = self._validate_data(X)
        
        responsibilities, _ = self.assignment_strategy.compute_assignments(
            X, self.representations, mixing_weights=self.weights_
        )
        
        return responsibilities
        
    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : Tensor of shape (n_samples,)
            Component labels
        """
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)
        
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        """Generate samples from fitted model.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        X : Tensor of shape (n_samples, n_features)
            Generated samples
        y : Tensor of shape (n_samples,)
            Component labels for each sample
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        # Sample component assignments
        component_dist = torch.distributions.Categorical(self.weights_)
        components = component_dist.sample((n_samples,))
        
        # Sample from each component
        samples = []
        for k in range(self.n_clusters):
            mask = components == k
            n_k = mask.sum().item()
            
            if n_k > 0:
                samples_k = self.representations[k].sample(n_k)
                samples.append((samples_k, mask))
                
        # Combine samples
        X = torch.zeros(n_samples, self.means_.shape[1], device=self.device)
        for samples_k, mask in samples:
            X[mask] = samples_k
            
        return X, components
        
    def score(self, X: Tensor, y: Optional[Tensor] = None) -> float:
        """Compute average log-likelihood.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to score
        y : Ignored
            
        Returns
        -------
        score : float
            Average log-likelihood
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        X = self._validate_data(X)
        
        # Compute log-likelihood
        log_probs = []
        for x in X:
            log_prob = torch.logsumexp(
                torch.stack([
                    rep.log_likelihood(x.unsqueeze(0)).squeeze() + torch.log(self.weights_[k])
                    for k, rep in enumerate(self.representations)
                ]), dim=0
            )
            log_probs.append(log_prob)
            
        return torch.stack(log_probs).mean().item()
        
    def bic(self, X: Tensor) -> float:
        """Bayesian Information Criterion.
        
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        
        Parameters
        ----------
        X : Tensor
            Data to evaluate
            
        Returns
        -------
        bic : float
            BIC score (lower is better)
        """
        n_samples = len(X)
        log_likelihood = self.score(X) * n_samples
        
        # Count parameters
        n_features = X.shape[1]
        
        # Means: n_clusters * n_features
        n_params = self.n_clusters * n_features
        
        # Covariances
        if self.covariance_type == 'full':
            # Symmetric matrix: n(n+1)/2 per cluster
            n_params += self.n_clusters * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diagonal':
            n_params += self.n_clusters * n_features
        elif self.covariance_type == 'spherical':
            n_params += self.n_clusters
        elif self.covariance_type == 'tied':
            n_params += n_features * (n_features + 1) // 2
            
        # Mixing weights (n_clusters - 1 due to constraint)
        n_params += self.n_clusters - 1
        
        return -2 * log_likelihood + n_params * torch.log(torch.tensor(n_samples)).item()
        
    def aic(self, X: Tensor) -> float:
        """Akaike Information Criterion.
        
        AIC = -2 * log_likelihood + 2 * n_params
        
        Parameters
        ----------
        X : Tensor
            Data to evaluate
            
        Returns
        -------
        aic : float
            AIC score (lower is better)
        """
        n_samples = len(X)
        log_likelihood = self.score(X) * n_samples
        
        # Same parameter counting as BIC
        n_features = X.shape[1]
        n_params = self.n_clusters * n_features
        
        if self.covariance_type == 'full':
            n_params += self.n_clusters * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diagonal':
            n_params += self.n_clusters * n_features
        elif self.covariance_type == 'spherical':
            n_params += self.n_clusters
        elif self.covariance_type == 'tied':
            n_params += n_features * (n_features + 1) // 2
            
        n_params += self.n_clusters - 1
        
        return -2 * log_likelihood + 2 * n_params