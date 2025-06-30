"""Assignment strategies for clustering algorithms."""

from .hard import HardAssignment, HardAssignmentWithInfo
from .penalized import PenalizedAssignment
from .soft import (
    SoftAssignment,
    GaussianSoftAssignment,
    AnnealedSoftAssignment,
    EntropyRegularizedAssignment
)

__all__ = [
    'HardAssignment',
    'HardAssignmentWithInfo',
    'PenalizedAssignment',
    'SoftAssignment',
    'GaussianSoftAssignment',
    'AnnealedSoftAssignment',
    'EntropyRegularizedAssignment'
]