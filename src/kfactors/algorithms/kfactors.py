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
from ..representations.eigenfactor import EigenFactorRepresentation
from ..assignments.factors import IndependentFactorAssignment
from ..updates.sequential_pca import SequentialPCAUpdater
from ..initialization.kmeans_plusplus import KMeansPlusPlusInit
from ..utils.convergence import ChangeInObjective


class KFactorsObjective(ClusteringObjective):
    """K-Factors objective: penalized reconstruction error."""
    
    def compute(self, points: Tensor, representations: List[ClusterRepresentation],
                assignments: Tensor, **kwargs) -> Tensor:
        """Compute sum of squared residuals after projection."""
        device, dtype = points.device, points.dtype
        total_error = 0.0

        # Prefer precomputed distances from the assignment step
        aux_info = kwargs.get('aux_info', {})
        distances = aux_info.get('distances', None)
        if distances is not None:
            if not torch.is_tensor(distances):
                distances = torch.as_tensor(distances, device=device, dtype=dtype)
            # Pick the chosen column per sample and sum → torch scalar
            return distances.gather(1, assignments.view(-1, 1)).sum()

        # Fallback: recompute distances
        total_error = torch.zeros((), device=device, dtype=dtype)

        # Flat EigenFactorRepresentation path
        try:
            from ..representations.eigenfactor import EigenFactorRepresentation
        except Exception:
            EigenFactorRepresentation = None  # type: ignore

        if (EigenFactorRepresentation is not None and
            len(representations) == 1 and
            isinstance(representations[0], EigenFactorRepresentation)):
            ef = representations[0]
            x  = points.unsqueeze(1)                # (n, 1, D)
            mu = ef.means.unsqueeze(0)              # (1, Kf, D)
            w  = ef.vectors.unsqueeze(0)            # (1, Kf, D)
            centered = x - mu                       # (n, Kf, D)
            num = (centered * w).sum(dim=2)         # (n, Kf)
            den = (ef.vectors * ef.vectors).sum(dim=1).clamp_min(1e-12)  # (Kf,)
            dist = (num * num) / den.unsqueeze(0)   # (n, Kf)
            return dist.gather(1, assignments.view(-1, 1)).sum()

        # Classic per-representation path
        for k, rep in enumerate(representations):
            mask = (assignments == k)
            if mask.any():
                total_error = total_error + rep.distance_to_point(points[mask]).sum()

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
    self.n_factors : int
        Number of principal components 
    init : str or array-like, default='k-means++'
        Initialization method
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
    labels_ : Tensor of shape (n_samples,)
        Final cluster assignments
    direction_tracker_ : DirectionTracker
        Tracks claimed directions per point
    n_iter_ : int
        Total iterations across all stages
    """
    
    def __init__(self,
                 n_factors: int,
                 init: str = 'k-means++',
                 max_iter: int = 30,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """Initialize K-Factors algorithm."""
        super().__init__(
            n_clusters=n_factors,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            device=device
        )
        
        self.n_factors = n_factors
        self.init = init
        
        # K-Factors specific attributes
        self.direction_tracker_ = None
        self.stage_history_ = []
        self.representations = []
        
    def _create_components(self) -> None:
        """Create K-Factors specific components."""
        # Assignment with penalty
        self.assignment_strategy = IndependentFactorAssignment()
        
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
            n_points, self.n_factors, self.device
        )
        
        # Get initial centers
        initial_centroids = self.initialization_strategy.initialize(
            data, self.n_factors
        )
        
        # Convert to Eigen Factors representation
        eigen_factors = EigenFactorRepresentation(
            dimension=dimension,
            n_factors=self.n_factors,
            device=self.device,
            init="zeros",
        )

        # Stack the K means and assign directly (no repeat)
        init_means = torch.stack(
            [rep.get_parameters()['mean'] for rep in initial_centroids]
        ).to(self.device)  # (K, D)
        eigen_factors.means = init_means  # shape matches: (K, D)

        # Vectors remain at their default small-random init inside EigenFactorRepresentation
        representations = [eigen_factors]
            
        print(f"_create_representations {representations[0].dump()}")
        return representations
        
    # FIXED FOR EF
    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> 'KFactors':
        """Fit K-Factors with a flat pool of factors (EigenFactorRepresentation)."""
        # Validate data
        X = self._validate_data(X)
        n_points, dimension = X.shape

        if self.verbose:
            print(f"K-Factors (flat): {self.n_factors} = "
                f"{self.n_factors} factors")

        # Create components
        self._create_components()

        # Initialize representations and direction tracker (now a single EigenFactorRepresentation)
        self.representations = self._create_representations(X)
        eigen_factors = self.representations[0]

        # Main sequential extraction loop
        self.n_iter_ = 0
        self.stage_history_ = []

        for stage in range(self.n_factors):
            if self.verbose:
                print(f"\n=== Stage {stage + 1}/{self.n_factors} ===")

            # Reset convergence for this stage
            self.convergence_criterion.reset()
            stage_converged = False

            # Track stage in direction tracker
            self.direction_tracker_.current_stage = stage

            # Inner loop for this stage
            for iter_in_stage in range(self.max_iter):
                print(f"Iteration {iter_in_stage}")
                # Assignment (penalized)
                assignments, aux_info = self.assignment_strategy.compute_assignments(
                    X,
                    self.representations,
                    direction_tracker=self.direction_tracker_,
                    current_stage=stage,
                )
                # print(f"assignments = {assignments}")
                # print(f"aux_info = {aux_info}")

                # Stage weights from penalties (one weight per sample for its assigned factor)
                penalties = aux_info.get('penalties', None)
                if penalties is not None:
                    stage_weights = penalties.gather(1, assignments.view(-1, 1)).squeeze(1)
                else:
                    stage_weights = torch.ones_like(assignments, dtype=torch.float32, device=X.device)
                # print(f"penalties = {penalties}")

                # Update all factors in one shot (flat representation)
                # Uses hard assignments + per-sample weights to compute each (mu_k, w_k)
                print(f"old model = {eigen_factors.dump()}")
                eigen_factors.update_from_points(
                    X,
                    weights=stage_weights,          # (n,)
                    assignments=assignments,        # (n,)
                )
                print(f"new model = {eigen_factors.dump()}")

                # Objective for logging / convergence
                objective_value = self.objective.compute(
                    X, self.representations, assignments, aux_info=aux_info
                )

                # Convergence check
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

            # Direction tracker: claim directions for the assigned factor of each point
            self.direction_tracker_.add_claimed_directions_batch(
                assignments,                  # (n,)
                eigen_factors.vectors,        # (K, D)
                claimed_weights=stage_weights # (n,)
            )

            # Stage bookkeeping
            # print(f"objective_value = {objective_value}")
            self.stage_history_.append({
                'stage': stage,
                'iterations': iter_in_stage + 1,
                'objective': objective_value.item(),
                'converged': stage_converged
            })

            if not stage_converged and self.verbose:
                warnings.warn(f"Stage {stage + 1} did not converge")

        # Final artifacts
        self.labels_ = assignments                      # final factor index per point
        self.cluster_means_ = eigen_factors.means.clone()     # (K, D)
        self.cluster_vectors_ = eigen_factors.vectors.clone() # (K, D)

        self.fitted_ = True

        if self.verbose:
            print(f"\nK-Factors completed: {self.n_iter_} total iterations")

        return self
    
    # FIXED FOR EF
    def transform(self, X: Tensor) -> Tensor:
        """Transform data to flat factor coordinates (scores).

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_transformed : Tensor of shape (n_samples, n_factors)
            Per-sample scores y_k = w_k^T (x - mu_k) for all factors.
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling transform")

        X = self._validate_data(X)                 # (n, D)
        eigen_factors = self.representations[0]    # EigenFactorRepresentation

        W = eigen_factors.vectors                  # (K, D)
        MU = eigen_factors.means                   # (K, D)
        b = -(W * MU).sum(dim=1)                   # (K,)

        # scores: (n, D) @ (D, K) -> (n, K), then add bias
        return X @ W.t() + b.unsqueeze(0)
        
    # FIXED FOR EF    
    def inverse_transform(self, X_transformed: Tensor, assignments: Optional[Tensor] = None) -> Tensor:
        """Reconstruct data from flat factor scores.

        This inverts y = W x + b (per sample) in least-squares sense:
            x ≈ W^+ (y - b)
        where W^+ is the Moore–Penrose pseudoinverse of W.

        Parameters
        ----------
        X_transformed : Tensor of shape (n_samples, n_factors)
            Scores returned by transform(), i.e., y_k = w_k^T x + b_k.
        assignments : Tensor, optional
            Ignored for flat factors (kept for backward compatibility).

        Returns
        -------
        X : Tensor of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first")

        # Validate and move to device
        Y = X_transformed
        if not isinstance(Y, Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        Y = Y.to(self.device)
        if Y.dim() != 2:
            raise ValueError(f"Expected 2D tensor for X_transformed, got {Y.dim()}D")

        # Grab flat factors
        eigen_factors = self.representations[0]  # EigenFactorRepresentation
        W = eigen_factors.vectors                # (K, D)
        MU = eigen_factors.means                 # (K, D)

        # Bias per factor: b_k = - w_k^T mu_k
        b = -(W * MU).sum(dim=1)                 # (K,)

        # Sanity check: columns must match number of factors
        n_factors = W.shape[0]
        if Y.shape[1] != n_factors:
            raise ValueError(f"X_transformed has {Y.shape[1]} columns, but model has {n_factors} factors")

        # Center scores and solve least squares in one shot for the batch:
        # X ≈ (Y - b)^T via pseudoinverse: X = (Y - b) @ (W^+)^T
        Yc = Y - b.unsqueeze(0)                  # (n, K)
        W_pinv = torch.linalg.pinv(W)            # (D, K)
        X_rec = Yc @ W_pinv.t()                  # (n, D)

        return X_rec
