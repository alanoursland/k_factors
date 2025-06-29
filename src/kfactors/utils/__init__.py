"""Utility functions for K-Factors algorithms."""

from .linalg import (
    orthogonalize_basis,
    safe_svd,
    safe_eigh,
    power_iteration_eigh,
    project_to_orthogonal_complement,
    gram_schmidt,
    matrix_sqrt,
    low_rank_approx,
    solve_linear_system
)

from .convergence import (
    ChangeInAssignments,
    ChangeInObjective,
    ParameterChange,
    CombinedCriterion,
    MaxIterations
)

__all__ = [
    # Linear algebra
    'orthogonalize_basis',
    'safe_svd',
    'safe_eigh',
    'power_iteration_eigh',
    'project_to_orthogonal_complement',
    'gram_schmidt',
    'matrix_sqrt',
    'low_rank_approx',
    'solve_linear_system',
    
    # Convergence criteria
    'ChangeInAssignments',
    'ChangeInObjective',
    'ParameterChange',
    'CombinedCriterion',
    'MaxIterations'
]