# eigenfactor.py
"""
EigenFactor representation: a flat pool of K eigen-like features.

Each feature k is an affine hyperplane parameterized by:
  - mu_k: (D,) local anchor (mean)
  - w_k:  (D,) direction vector (not forced unit or orthogonal)

Natural score for a point x:
  y_k(x) = w_k^T (x - mu_k) = w_k^T x + b_k,  where b_k = - w_k^T mu_k.

This class is intentionally *not* a "cluster representation". It holds a
global dictionary of K factors and exposes:
  - compute_scores(x): (n, K) scores y_k(x)
  - distance_to_point(x, indices=None): per-point squared distance *to a single
    feature's hyperplane* (if indices provided) or the minimum across features.

Update rule (when assignments are given):
  - For each factor k:
      mu_k := (weighted mean of assigned points)
      w_k  := top eigenvector of local (weighted) covariance, scaled by sqrt(lambda_top)
              so that ||w_k|| contains variance magnitude.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor

from .base_representation import BaseRepresentation


class EigenFactorRepresentation(BaseRepresentation):
    """
    Flat dictionary of K affine eigen-like features in D dimensions.

    Parameters
    ----------
    dimension : int
        Ambient dimension D.
    n_factors : int
        Number of features (K).
    device : torch.device
        Storage / compute device.
    init : str, optional
        Initialization scheme for (means, vectors). Currently supports:
          - 'zeros' means + random normal vectors
    eps : float, optional
        Numerical epsilon used in normalizations and safeguards.
    """

    def __init__(
        self,
        dimension: int,
        n_factors: int,
        device: torch.device,
        init: str = "zeros",
        eps: float = 1e-12,
    ):
        # NOTE: BaseRepresentation will create a single self._mean (D,)
        # which isn't used as "the" mean here (we manage per-factor means).
        super().__init__(dimension, device)

        if n_factors <= 0:
            raise ValueError("n_factors must be positive")

        self.n_factors = n_factors
        self.eps = float(eps)

        # Storage: K x D tensors
        if init == "zeros":
            self._means = torch.zeros(n_factors, dimension, device=device)
            # random small vectors; no orthogonality or unit constraint
            self._vectors = torch.randn(n_factors, dimension, device=device) * 0.01
        else:
            raise ValueError(f"Unknown init scheme: {init}")

    # -------------------------------
    # Core tensors (means, vectors)
    # -------------------------------
    @property
    def means(self) -> Tensor:
        """(K, D) means μ_k."""
        return self._means

    @means.setter
    def means(self, value: Tensor):
        assert value.shape == (self.n_factors, self._dimension)
        self._means = value.to(self._device)

    @property
    def vectors(self) -> Tensor:
        """(K, D) vectors w_k (scale encodes variance; no unit/orthogonal constraint)."""
        return self._vectors

    @vectors.setter
    def vectors(self, value: Tensor):
        assert value.shape == (self.n_factors, self._dimension)
        self._vectors = value.to(self._device)

    @property
    def biases(self) -> Tensor:
        """(K,) biases b_k = - w_k^T μ_k for linear forms y = w^T x + b."""
        # (K, D) · (K, D) -> (K,) via rowwise dot
        return -(self._vectors * self._means).sum(dim=1)

    # --------------------------------------
    # Scoring & distances for convenience
    # --------------------------------------
    def compute_scores(self, points: Tensor) -> Tensor:
        """
        Compute all feature scores y_k(x) = w_k^T (x - mu_k) for a batch.

        Args
        ----
        points : (n, D)

        Returns
        -------
        scores : (n, K)
        """
        self._check_points_shape(points)
        # y = X W^T + b, with b = -w·mu (per-row)
        # XW^T: (n, D) @ (D, K) -> (n, K)
        WX = points @ self._vectors.t()                    # (n, K)
        b = self.biases                                    # (K,)
        return WX + b.unsqueeze(0)

    def distance_to_point(self, points: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        """
        Squared (signed) distance to hyperplanes, normalized by ||w_k||^2.

        If `indices` is provided (shape (n,)), return the distance to the
        corresponding feature per point:
            d_i = [ w_{k_i}^T (x_i - mu_{k_i}) ]^2 / (||w_{k_i}||^2 + eps)

        If `indices` is None, returns the minimum distance across all factors
        for each point:
            d_i = min_k [ (w_k^T (x_i - mu_k))^2 / (||w_k||^2 + eps) ]

        Returns
        -------
        (n,) tensor of distances.
        """
        self._check_points_shape(points)
        n, d = points.shape
        K = self.n_factors
        eps = self.eps

        if indices is not None:
            # Per-point chosen factor
            assert indices.shape == (n,)
            # Gather μ_k and w_k for each point
            mu = self._means[indices]          # (n, D)
            w = self._vectors[indices]         # (n, D)
            centered = points - mu             # (n, D)
            num = (centered * w).sum(dim=1)    # (n,)
            den = (w * w).sum(dim=1).clamp_min(eps)
            return (num * num) / den

        # Otherwise, compute all distances then min across k
        # Expand to (n, K, D): x_i - mu_k
        x_exp = points.unsqueeze(1)                       # (n, 1, D)
        mu_all = self._means.unsqueeze(0)                 # (1, K, D)
        centered = x_exp - mu_all                         # (n, K, D)

        # Dot with w_k per k: (n, K)
        w_all = self._vectors.unsqueeze(0)                # (1, K, D)
        num = (centered * w_all).sum(dim=2)               # (n, K)

        den = (self._vectors * self._vectors).sum(dim=1)  # (K,)
        den = den.clamp_min(eps).unsqueeze(0)             # (1, K)

        dists = (num * num) / den                         # (n, K)
        return dists.min(dim=1).values                    # (n,)

    # --------------------------------------
    # Updating from data + (soft) assignments
    # --------------------------------------
    def update_from_points(
        self,
        points: Tensor,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Update all features from data under (hard or soft) assignments.

        Supported kwargs:
          - assignments: (n,) long tensor of factor indices k in [0..K-1]
          - soft_assignments: (n, K) responsibilities in [0,1], rows sum to 1

        Notes
        -----
        - If both are provided, soft_assignments takes precedence.
        - If neither is provided, does nothing (no notion of "assigned" points).
        - For each factor k:
            mu_k := weighted mean (by r_{ik})
            w_k  := top eigenvector of weighted covariance, scaled by sqrt(lambda_top)
        """
        self._check_points_shape(points)
        n, d = points.shape
        K = self.n_factors
        device = points.device
        eps = self.eps

        soft = kwargs.get("soft_assignments", None)
        hard = kwargs.get("assignments", None)

        if soft is None and hard is None:
            return  # nothing to update against

        if soft is not None:
            assert soft.shape == (n, K)
            R = soft.to(device).clamp_min(0.0)
        else:
            # hard assignments → one-hot soft
            assert hard.shape == (n,)
            R = torch.zeros(n, K, device=device, dtype=points.dtype)
            R.scatter_(1, hard.view(-1, 1), 1.0)

        # For numerical stability, allow an overall per-sample weight, if provided.
        if weights is not None:
            assert weights.shape == (n,)
            w_samples = weights.to(device).clamp_min(0.0).unsqueeze(1)  # (n,1)
            R = R * w_samples

        # Update each factor independently
        for k in range(K):
            r_k = R[:, k]                             # (n,)
            mass = r_k.sum()
            if mass.item() <= eps:
                # no effective responsibility for this factor; skip
                continue

            # Weighted mean
            alpha = (r_k / (mass + eps)).unsqueeze(1)  # (n,1)
            mu_k = (alpha * points).sum(dim=0)         # (D,)
            self._means[k] = mu_k

            # Centered with sqrt-weights for covariance
            centered = points - mu_k.unsqueeze(0)      # (n, D)
            cw = centered * torch.sqrt(r_k + eps).unsqueeze(1)

            # Covariance (scaled by mass)
            # S = (cw^T cw) / mass
            # Top eigvec/value of S:
            try:
                # Use SVD on cw: cw = U S V^T  ⇒ top right singular vector is principal dir
                # Singular values^2 / mass are eigenvalues of S
                U, S, Vt = torch.linalg.svd(cw, full_matrices=False)
                v_top = Vt[0]                      # (D,)
                lambda_top = (S[0] ** 2) / (mass + eps)
            except Exception:
                # Fallback via eigh on (D,D)
                S_cov = (cw.t() @ cw) / (mass + eps)   # (D,D)
                evals, evecs = torch.linalg.eigh(S_cov)
                idx = torch.argmax(evals)
                v_top = evecs[:, idx]
                lambda_top = torch.clamp(evals[idx], min=eps)

            # Scale direction by sqrt(lambda_top) so ||w|| carries variance magnitude.
            w_k = v_top * torch.sqrt(torch.clamp(lambda_top, min=eps))
            self._vectors[k] = w_k

    # --------------------------------------
    # BaseRepresentation API glue
    # --------------------------------------
    def get_parameters(self) -> Dict[str, Tensor]:
        """Return {'means': (K,D), 'vectors': (K,D)}."""
        return {
            "means": self._means.clone(),
            "vectors": self._vectors.clone(),
        }

    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set state from dict (expects shapes (K,D))."""
        if "means" in params:
            self.means = params["means"]
        if "vectors" in params:
            self.vectors = params["vectors"]

    def to(self, device: torch.device) -> "EigenFactorRepresentation":
        """Device move (override BaseRepresentation.to to handle (K,D) tensors)."""
        new_repr = EigenFactorRepresentation(
            self._dimension,
            self.n_factors,
            device=device,
            init="zeros",
            eps=self.eps,
        )
        params = self.get_parameters()
        params = {k: v.to(device) for k, v in params.items()}
        new_repr.set_parameters(params)
        return new_repr

    def dump(self) -> str:
        return ("EigenFactorRepresentation\n"
                f"  dimension={self._dimension},\n" +
                f"  n_factors={self.n_factors},\n" +
                f"  means={self.means},\n" +
                f"  vectors={self.vectors},\n" +
                f"")

    def __repr__(self) -> str:
        return (f"EigenFactorRepresentation(dimension={self._dimension}, "
                f"n_factors={self.n_factors})")
