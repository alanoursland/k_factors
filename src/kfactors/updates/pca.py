"""
PCA update strategy for subspace clustering.

Updates cluster representations by performing PCA on assigned points.
"""

from typing import Optional
import torch
from torch import Tensor

from ..base.interfaces import ParameterUpdater, ClusterRepresentation
from ..representations.subspace import SubspaceRepresentation
from ..representations.ppca import PPCARepresentation
from ..utils.linalg import safe_svd


class PCAUpdater(ParameterUpdater):
    """Updates cluster subspace using PCA on assigned points.
    
    Extracts the top principal components to form an orthonormal basis
    for the cluster's subspace.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Args:
            n_components: Number of components to extract.
                         If None, uses the representation's dimension.
        """
        self.n_components = n_components
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignments: Optional[Tensor] = None,
               **kwargs) -> None:
        """Update representation using PCA.
        
        Args:
            representation: Subspace or PPCA representation to update
            points: Points assigned to this cluster
            assignments: Weights for soft assignment (optional)
            **kwargs: Additional parameters (unused)
        """
        if len(points) == 0:
            return
            
        # Determine number of components
        if isinstance(representation, SubspaceRepresentation):
            n_components = self.n_components or representation.subspace_dim
        elif isinstance(representation, PPCARepresentation):
            n_components = self.n_components or representation.latent_dim
        else:
            raise TypeError(f"PCA updater requires Subspace or PPCA representation, "
                          f"got {type(representation)}")
                          
        # Update mean
        if assignments is None:
            # Hard assignment - simple mean
            new_mean = points.mean(dim=0)
            centered = points - new_mean
        else:
            # Soft assignment - weighted mean
            assert assignments.shape == (points.shape[0],)
            total_weight = assignments.sum()
            
            if total_weight <= 0:
                return
                
            normalized_weights = assignments / total_weight
            new_mean = torch.sum(points * normalized_weights.unsqueeze(1), dim=0)
            centered = points - new_mean
            
            # Apply sqrt of weights for weighted PCA
            centered = centered * torch.sqrt(normalized_weights).unsqueeze(1)
            
        representation.mean = new_mean
        
        if n_components == 0:
            return  # No subspace to update
            
        # Handle degenerate cases
        n_points, dimension = centered.shape
        max_components = min(n_points - 1, dimension, n_components)
        
        if max_components <= 0:
            # Not enough points for PCA
            if isinstance(representation, SubspaceRepresentation):
                representation._basis = torch.zeros(dimension, 0, device=points.device)
            return
            
        # Perform SVD for PCA
        try:
            # Use economy SVD for efficiency
            U, S, Vt = safe_svd(centered, full_matrices=False)
            V = Vt.t()
            
            # Extract top components
            actual_components = min(max_components, V.shape[1])
            basis = V[:, :actual_components]
            
            # Pad with zeros if needed
            if actual_components < n_components:
                padding = torch.zeros(dimension, n_components - actual_components, 
                                    device=points.device)
                basis = torch.cat([basis, padding], dim=1)
                
        except Exception as e:
            import warnings
            warnings.warn(f"SVD failed in PCA update: {e}. Using identity basis.")
            
            # Fallback to identity basis
            basis = torch.eye(dimension, n_components, device=points.device)
            
        # Update representation
        if isinstance(representation, SubspaceRepresentation):
            representation.basis = basis
        elif isinstance(representation, PPCARepresentation):
            # For PPCA, also need to scale by singular values
            if len(S) >= n_components:
                # Scale basis by sqrt of eigenvalues
                scales = S[:n_components] / torch.sqrt(torch.tensor(n_points - 1, dtype=torch.float32))
                representation.W = basis * scales.unsqueeze(0)
                
                # Update noise variance from remaining eigenvalues
                if len(S) > n_components:
                    remaining_variance = torch.mean(S[n_components:] ** 2) / (n_points - 1)
                    representation.variance = remaining_variance
                else:
                    representation.variance = torch.tensor(1e-6, device=points.device)
            else:
                # Not enough singular values
                representation.W = basis
                representation.variance = torch.tensor(1e-6, device=points.device)


class IncrementalPCAUpdater(ParameterUpdater):
    """Incremental PCA updater for streaming data.
    
    Updates the subspace incrementally without storing all points,
    useful for large datasets or online learning.
    """
    
    def __init__(self, n_components: int, learning_rate: float = 0.1):
        """
        Args:
            n_components: Number of components
            learning_rate: Learning rate for incremental updates
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self._n_samples_seen = {}  # Track samples per cluster
        
    def update(self, representation: ClusterRepresentation,
               points: Tensor,
               assignments: Optional[Tensor] = None,
               **kwargs) -> None:
        """Update representation incrementally.
        
        Uses a variant of Oja's rule for online PCA.
        """
        if len(points) == 0:
            return
            
        cluster_id = id(representation)  # Unique identifier
        
        # Initialize sample counter
        if cluster_id not in self._n_samples_seen:
            self._n_samples_seen[cluster_id] = 0
            
        n_seen = self._n_samples_seen[cluster_id]
        
        # Update mean incrementally
        old_mean = representation.mean.clone()
        
        if assignments is None:
            # Hard assignment
            batch_size = len(points)
            batch_mean = points.mean(dim=0)
            
            # Incremental mean update
            alpha = batch_size / (n_seen + batch_size)
            new_mean = (1 - alpha) * old_mean + alpha * batch_mean
            
        else:
            # Soft assignment
            total_weight = assignments.sum()
            if total_weight <= 0:
                return
                
            normalized_weights = assignments / total_weight
            batch_mean = torch.sum(points * normalized_weights.unsqueeze(1), dim=0)
            
            # Weight-aware incremental update
            alpha = total_weight / (n_seen + total_weight)
            new_mean = (1 - alpha) * old_mean + alpha * batch_mean
            
        representation.mean = new_mean
        
        # Center points
        centered = points - new_mean
        
        # Get current basis
        if isinstance(representation, SubspaceRepresentation):
            if representation.basis.shape[1] == 0:
                # Initialize with random orthonormal basis
                init_basis = torch.randn(representation.dimension, self.n_components,
                                       device=points.device)
                representation.basis, _ = torch.linalg.qr(init_basis)
                
            basis = representation.basis
        else:
            raise TypeError("Incremental PCA requires SubspaceRepresentation")
            
        # Incremental update using Oja's rule variant
        for i, point in enumerate(centered):
            if assignments is not None:
                weight = assignments[i]
            else:
                weight = 1.0
                
            # Project onto current basis
            coeffs = torch.matmul(point, basis)
            
            # Reconstruction
            recon = torch.matmul(coeffs, basis.t())
            
            # Error
            error = point - recon
            
            # Update basis
            lr = self.learning_rate * weight / (n_seen + i + 1) ** 0.5
            basis = basis + lr * torch.outer(error, coeffs)
            
            # Orthonormalize periodically
            if (i + 1) % 10 == 0:
                basis, _ = torch.linalg.qr(basis)
                
        # Final orthonormalization
        basis, _ = torch.linalg.qr(basis)
        representation.basis = basis
        
        # Update sample counter
        if assignments is None:
            self._n_samples_seen[cluster_id] += len(points)
        else:
            self._n_samples_seen[cluster_id] += assignments.sum().item()