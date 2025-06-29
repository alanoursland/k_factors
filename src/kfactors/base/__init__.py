"""Base classes and interfaces for K-Factors clustering algorithms."""

from .interfaces import (
    ClusterRepresentation,
    AssignmentStrategy,
    ParameterUpdater,
    DistanceMetric,
    InitializationStrategy,
    ConvergenceCriterion,
    ClusteringObjective
)

from .data_structures import (
    ClusterState,
    AssignmentMatrix,
    DirectionTracker,
    AlgorithmState
)

from .clustering_base import BaseClusteringAlgorithm

__all__ = [
    # Interfaces
    'ClusterRepresentation',
    'AssignmentStrategy', 
    'ParameterUpdater',
    'DistanceMetric',
    'InitializationStrategy',
    'ConvergenceCriterion',
    'ClusteringObjective',
    
    # Data structures
    'ClusterState',
    'AssignmentMatrix',
    'DirectionTracker',
    'AlgorithmState',
    
    # Base algorithm
    'BaseClusteringAlgorithm'
]