"""
C-Factors (Mixture of Probabilistic PCA) clustering.

Soft clustering algorithm that models each cluster as a low-rank Gaussian
(PPCA model). This is the soft counterpart to K-Factors.
"""

from typing import Optional, List, Union
import torch
from torch import Tensor

from ..base.clustering_base import BaseClusteringAlgorithm
from ..base.interfaces import ClusterRepresentation, ClusteringObjective
from ..representations.ppca import PPCARepresentation
from ..assignments.soft import SoftAssignment
from ..updates.em_updates import PPCAEMUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective


class CFactorsObjective(ClusteringObjective):
    """C-Factors objective: negative log-likelihood under mixture of PPCA."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor, **kwargs) -> Tensor:
        """Compute negative log-likelihood."""
        aux_info = kwargs.get('aux_info', {})
        
        # Use pre-computed log-likelihood from E-step
        if 'total_log_likelihood' in aux_info:
            return -aux_info['total_log_likelihood']
            
        # Fallback computation
        log_marginal = aux_info.get('log_marginal', None)
        if log_marginal is not None:
            return -log_marginal.sum()
            
        # Manual computation
        n_clusters = len(representations)
        mixing_weights = aux_info.get('mixing_weights',
                                    torch.ones(n_clusters) / n_clusters)
        
        log_likelihood = 0.0
        for i, point in enumerate(points):
            point_ll = torch.logsumexp(
                torch.stack([
                    rep.log_likelihood(point.unsqueeze(0)).squeeze() + 
                    torch.log(mixing_weights[k])
                    for k, rep in enumerate(representations)
                ]), dim=0
            )
            log_likelihood += point_ll
            
        return -log_likelihood
        
    @property
    def minimize(self) -> bool:
        return True


class CFactors(BaseClusteringAlgorithm):
    """C-Factors (Mixture of PPCA) clustering algorithm.
    
    C-Factors is the soft-assignment version of K-Factors, using EM
    to fit a mixture of Probabilistic PCA models. Each cluster is
    represented by a low-rank Gaussian with covariance WW^T + σ²I.
    
    This is equivalent to the classical Mixture of Factor Analyzers
    or Mixture of PPCA, renamed here for pedagogical symmetry with
    K-Factors (as C-Means relates to K-Means).
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters/components
    n_components : int
        Latent dimension per cluster
    init : str or array-like, default='k-means++'
        Initialization method
    max_iter : int, default=100
        Maximum EM iterations
    tol : float, default=1e-4
        Convergence tolerance on log-likelihood
    reg_variance : float, default=1e-6
        Minimum noise variance for stability
    n_init : int, default=1
        Number of initialization attempts
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed
    device : torch.device, optional
        Computation device
        
    Attributes
    ----------
    cluster_centers_ : Tensor of shape (n_clusters, n_features)
        Cluster means
    factor_loadings_ : Tensor of shape (n_clusters, n_features, n_components)
        Factor loading matrices W for each cluster
    noise_variances_ : Tensor of shape (n_clusters,)
        Isotropic noise variance for each cluster
    weights_ : Tensor of shape (n_clusters,)
        Mixing weights
    responsibilities_ : Tensor of shape (n_samples, n_clusters)
        Final posterior probabilities
    lower_bound_ : float
        Final log-likelihood
    n_iter_ : int
        Number of iterations run
        
    Examples
    --------
    >>> import torch
    >>> from kfactors.algorithms import CFactors
    >>> 
    >>> # Data with local low-dimensional structure
    >>> X = torch.randn(500, 20)
    >>> 
    >>> # Fit C-Factors with 3D latent spaces
    >>> cfactors = CFactors(n_clusters=4, n_components=3)
    >>> cfactors.fit(X)
    >>> 
    >>> # Get soft assignments
    >>> probs = cfactors.predict_proba(X)
    >>> 
    >>> # Transform to latent space
    >>> Z = cfactors.transform(X)
    """
    
    def __init__(self,
                 n_clusters: int,
                 n_components: int,
                 init: Union[str, Tensor] = 'k-means++',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 reg_variance: float = 1e-6,
                 n_init: int = 1,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize C-Factors algorithm."""
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
        self.reg_variance = reg_variance
        self.n_init = n_init
        
        # C-Factors specific attributes
        self.cluster_centers_ = None
        self.factor_loadings_ = None
        self.noise_variances_ = None
        self.weights_ = None
        self.responsibilities_ = None
        self.lower_bound_ = -float('inf')
        
    def _create_components(self) -> None:
        """Create C-Factors specific components."""
        # Soft assignment for E-step
        self.assignment_strategy = SoftAssignment()
        
        # PPCA EM updater for M-step
        self.update_strategy = PPCAEMUpdater(
            n_components=self.n_components,
            min_variance=self.reg_variance
        )
        
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
            from ..initialization.from_previous import FromPreviousInit
            self.initialization_strategy = FromPreviousInit(self.init)
            
        # Convergence based on log-likelihood change
        self.convergence_criterion = ChangeInObjective(
            rel_tol=self.tol,
            abs_tol=1e-9,
            patience=2
        )
        
        # Objective
        self.objective = CFactorsObjective()
        
    def _create_representations(self, data: Tensor) -> List[ClusterRepresentation]:
        """Create PPCA representations."""
        representations = []
        dimension = data.shape[1]
        
        # Validate latent dimension
        if self.n_components >= dimension:
            raise ValueError(f"Latent dimension {self.n_components} must be less than "
                           f"data dimension {dimension}")
                           
        # Initialize
        init_representations = self.initialization_strategy.initialize(
            data, self.n_clusters
        )
        
        for init_rep in init_representations:
            ppca = PPCARepresentation(
                dimension=dimension,
                latent_dim=self.n_components,
                device=self.device,
                init_variance=1.0
            )
            ppca.mean = init_rep.get_parameters()['mean']
            representations.append(ppca)
            
        # Initialize mixing weights
        self.weights_ = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
        
        return representations
        
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'CFactors':
        """Fit C-Factors model using EM.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used
            
        Returns
        -------
        self : CFactors
            Fitted model
        """
        X = self._validate_data(X)
        
        if self.verbose:
            print(f"C-Factors: {self.n_clusters} clusters with "
                  f"{self.n_components}D latent spaces")
                  
        # Multiple initializations
        best_log_likelihood = -float('inf')
        best_state = None
        
        for init_iter in range(self.n_init):
            if self.verbose and self.n_init > 1:
                print(f"\nInitialization {init_iter + 1}/{self.n_init}")
                
            # Run EM
            self._single_fit(X)
            
            # Check if best so far
            if self.lower_bound_ > best_log_likelihood:
                best_log_likelihood = self.lower_bound_
                best_state = self._extract_state()
                
        # Use best initialization
        if best_state is not None:
            self._set_state(best_state)
            self.lower_bound_ = best_log_likelihood
            
        self.fitted_ = True
        return self
        
    def _single_fit(self, X: Tensor) -> None:
        """Single run of EM algorithm."""
        # Create components
        self._create_components()
        self.representations = self._create_representations(X)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            self.responsibilities_, aux_info = self.assignment_strategy.compute_assignments(
                X, self.representations, mixing_weights=self.weights_
            )
            
            # M-step: update parameters
            for k, ppca in enumerate(self.representations):
                resp_k = self.responsibilities_[:, k]
                self.update_strategy.update(ppca, X, resp_k)
                
            # Update mixing weights
            self.weights_ = self.responsibilities_.sum(dim=0) / len(X)
            self.weights_ = torch.clamp(self.weights_, min=1e-10)
            self.weights_ = self.weights_ / self.weights_.sum()
            
            # Compute objective
            objective = self.objective.compute(
                X, self.representations, self.responsibilities_, 
                aux_info=aux_info
            )
            
            # Check convergence
            converged = self.convergence_criterion.check({
                'iteration': iteration,
                'objective': objective.item()
            })
            
            if self.verbose >= 2:
                print(f"  Iteration {iteration:3d}: "
                      f"log-likelihood = {-objective.item():.6f}")
                      
            if converged:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            self.n_iter_ = iteration + 1
            
        self.lower_bound_ = -objective.item()
        
    def _extract_state(self) -> Dict[str, Any]:
        """Extract current model state."""
        self.cluster_centers_ = torch.stack([
            rep.mean for rep in self.representations
        ])
        
        self.factor_loadings_ = torch.stack([
            rep.W for rep in self.representations
        ])
        
        self.noise_variances_ = torch.stack([
            rep.variance for rep in self.representations
        ])
        
        return {
            'centers': self.cluster_centers_.clone(),
            'loadings': self.factor_loadings_.clone(),
            'variances': self.noise_variances_.clone(),
            'weights': self.weights_.clone(),
            'responsibilities': self.responsibilities_.clone()
        }
        
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set model state."""
        self.cluster_centers_ = state['centers']
        self.factor_loadings_ = state['loadings']
        self.noise_variances_ = state['variances']
        self.weights_ = state['weights']
        self.responsibilities_ = state['responsibilities']
        
        # Update representations
        for k, rep in enumerate(self.representations):
            rep.mean = self.cluster_centers_[k]
            rep.W = self.factor_loadings_[k]
            rep.variance = self.noise_variances_[k]
            
    def predict_proba(self, X: Tensor) -> Tensor:
        """Predict posterior probabilities.
        
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
            Cluster labels (hard assignment)
        """
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)
        
    def transform(self, X: Tensor) -> Tensor:
        """Transform data to latent space.
        
        For mixture models, this returns the concatenated latent
        representations weighted by responsibilities.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        Z : Tensor of shape (n_samples, n_clusters * n_components)
            Latent representations
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        X = self._validate_data(X)
        
        # Get responsibilities
        resp = self.predict_proba(X)
        
        # Transform each point using all clusters
        Z = torch.zeros(len(X), self.n_clusters * self.n_components,
                       device=self.device)
        
        for k, ppca in enumerate(self.representations):
            # Compute posterior mean for latent variables
            for i, x in enumerate(X):
                z_mean, _ = ppca.posterior_mean_cov(x)
                start_idx = k * self.n_components
                end_idx = (k + 1) * self.n_components
                
                # Weight by responsibility
                Z[i, start_idx:end_idx] = z_mean * resp[i, k]
                
        return Z
        
    def score_samples(self, X: Tensor) -> Tensor:
        """Compute log-likelihood of each sample.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : Tensor of shape (n_samples,)
            Log-likelihood of each sample
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")
            
        X = self._validate_data(X)
        
        log_probs = []
        for x in X:
            log_prob = torch.logsumexp(
                torch.stack([
                    rep.log_likelihood(x.unsqueeze(0)).squeeze() + 
                    torch.log(self.weights_[k])
                    for k, rep in enumerate(self.representations)
                ]), dim=0
            )
            log_probs.append(log_prob)
            
        return torch.stack(log_probs)
        
    def score(self, X: Tensor, y: Optional[Tensor] = None) -> float:
        """Average log-likelihood.
        
        Parameters
        ----------
        X : Tensor
            Data to score
        y : Ignored
            
        Returns
        -------
        score : float
            Average log-likelihood
        """
        return self.score_samples(X).mean().item()