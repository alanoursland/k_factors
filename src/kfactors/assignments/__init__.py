"""Assignment strategies for clustering algorithms."""

from .hard import HardAssignment, HardAssignmentWithInfo
from .penalized import PenalizedAssignment
from .soft import (
    SoftAssignment,
    GaussianSoftAssignment,
    AnnealedSoftAssignment,
    EntropyRegularizedAssignment
)
from .fuzzy import (
    FuzzyAssignment,
    PossibilisticAssignment,
    FuzzyPossibilisticAssignment
)

__all__ = [
    # Hard assignments
    'HardAssignment',
    'HardAssignmentWithInfo',
    'PenalizedAssignment',
    
    # Soft assignments
    'SoftAssignment',
    'GaussianSoftAssignment',
    'AnnealedSoftAssignment',
    'EntropyRegularizedAssignment',
    
    # Fuzzy assignments
    'FuzzyAssignment',
    'PossibilisticAssignment',
    'FuzzyPossibilisticAssignment'
]