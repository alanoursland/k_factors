"""
Initialization from previous solution or custom centers.

Useful for warm starts or when you have good initial guesses.
"""

from typing import List, Union
import torch
from torch import Tensor

from ..base.interfaces import InitializationStrategy, ClusterRepresentation
from ..representations.centroid import CentroidRepresentation
from ..base.data_structures import ClusterState


class FromPreviousInit(InitializationStrategy):
    """Initialize from previous cluster centers or custom starting points.
    
    Accepts either:
    - A tensor of shape (n_clusters, dimension) with initial centers
    - A ClusterState object from a previous run
    - A list of ClusterRepresentation objects
    """
    
    def __init__(self, initial_state: Union[Tensor, ClusterState, List[ClusterRepresentation]]):
        """
        Args:
            initial_state: Previous solution to use for initialization
        """
        self.initial_state = initial_state
        
    def initialize(self, points: Tensor, n_clusters: int,
                  **kwargs) -> List[ClusterRepresentation]:
        """Initialize from previous state.
        
        Args:
            points: (n, d) data points (used for validation)
            n_clusters: Expected number of clusters
            
        Returns:
            List of initialized representations
        """
        dimension = points.shape[1]
        device = points.device
        
        if isinstance(self.initial_state, Tensor):
            # Tensor of centers
            centers = self.initial_state.to(device)
            
            if centers.shape[0] != n_clusters:
                raise ValueError(f"Initial centers has {centers.shape[0]} clusters, "
                               f"but n_clusters={n_clusters}")
            if centers.shape[1] != dimension:
                raise ValueError(f"Initial centers has dimension {centers.shape[1]}, "
                               f"but data has dimension {dimension}")
                               
            representations = []
            for k in range(n_clusters):
                rep = CentroidRepresentation(dimension, device)
                rep.mean = centers[k].clone()
                representations.append(rep)
                
        elif isinstance(self.initial_state, ClusterState):
            # ClusterState object
            state = self.initial_state.to(device)
            
            if state.n_clusters != n_clusters:
                raise ValueError(f"ClusterState has {state.n_clusters} clusters, "
                               f"but n_clusters={n_clusters}")
                               
            representations = []
            for k in range(n_clusters):
                rep = CentroidRepresentation(dimension, device)
                rep.mean = state.means[k].clone()
                representations.append(rep)
                
        elif isinstance(self.initial_state, list):
            # List of representations
            if len(self.initial_state) != n_clusters:
                raise ValueError(f"Provided {len(self.initial_state)} representations, "
                               f"but n_clusters={n_clusters}")
                               
            representations = []
            for rep in self.initial_state:
                # Clone to new device if needed
                new_rep = rep.to(device)
                representations.append(new_rep)
                
        else:
            raise TypeError(f"Unknown initial_state type: {type(self.initial_state)}")
            
        return representations