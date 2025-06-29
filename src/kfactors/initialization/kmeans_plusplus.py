"""
K-means++ initialization strategy.

Selects initial cluster centers using the K-means++ algorithm, which chooses
centers that are far apart to improve convergence speed and quality.
"""

from typing import List, Optional
import torch
from torch import Tensor

from ..base.interfaces import InitializationStrategy, ClusterRepresentation
from ..representations.centroid import CentroidRepresentation


class KMeansPlusPlusInit(InitializationStrategy):
    """K-means++ initialization for better starting positions.
    
    Algorithm:
    1. Choose first center uniformly at random
    2. For each remaining center:
       - Compute distance from each point to nearest existing center
       - Choose next center with probability proportional to squared distance
    """
    
    def __init__(self, n_local_trials: Optional[int] = None):
        """
        Args:
            n_local_trials: Number of candidates to try for each center.
                           If None, uses 2 + log(k) as in sklearn
        """
        self.n_local_trials = n_local_trials
        
    def initialize(self, points: Tensor, n_clusters: int,
                  **kwargs) -> List[ClusterRepresentation]:
        """Initialize cluster centers using K-means++.
        
        Args:
            points: (n, d) data points
            n_clusters: Number of clusters
            
        Returns:
            List of initialized CentroidRepresentations
        """
        n_points, dimension = points.shape
        device = points.device
        
        if n_clusters > n_points:
            raise ValueError(f"Cannot create {n_clusters} clusters from {n_points} points")
            
        # Number of candidates to try per iteration
        if self.n_local_trials is None:
            n_local_trials = 2 + int(torch.log(torch.tensor(n_clusters, dtype=torch.float32)).item())
        else:
            n_local_trials = self.n_local_trials
            
        # Initialize list of centers
        centers = []
        center_indices = []
        
        # Choose first center uniformly at random
        first_idx = torch.randint(n_points, (1,), device=device).item()
        centers.append(points[first_idx].clone())
        center_indices.append(first_idx)
        
        # Compute initial distances
        distances = torch.sum((points - centers[0].unsqueeze(0)) ** 2, dim=1)
        
        # Choose remaining centers
        for c in range(1, n_clusters):
            # Compute probabilities proportional to squared distances
            probabilities = distances / distances.sum()
            
            # Sample candidates
            candidates_idx = torch.multinomial(probabilities, n_local_trials, replacement=True)
            
            # For each candidate, compute its potential (sum of min distances if chosen)
            best_potential = float('inf')
            best_candidate = None
            
            for idx in candidates_idx:
                # Compute distances from all points to this candidate
                candidate_distances = torch.sum((points - points[idx].unsqueeze(0)) ** 2, dim=1)
                
                # Update distances (take minimum with existing)
                new_distances = torch.minimum(distances, candidate_distances)
                
                # Compute potential (sum of squared distances)
                potential = new_distances.sum().item()
                
                if potential < best_potential:
                    best_potential = potential
                    best_candidate = idx.item()
                    
            # Add best candidate as new center
            centers.append(points[best_candidate].clone())
            center_indices.append(best_candidate)
            
            # Update distances
            new_center_distances = torch.sum((points - centers[-1].unsqueeze(0)) ** 2, dim=1)
            distances = torch.minimum(distances, new_center_distances)
            
        # Create CentroidRepresentations
        representations = []
        for center in centers:
            rep = CentroidRepresentation(dimension, device)
            rep.mean = center
            representations.append(rep)
            
        return representations