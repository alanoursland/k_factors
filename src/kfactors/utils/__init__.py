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

from .metrics import (
    # Distance functions
    pairwise_distances,
    
    # Internal metrics (no ground truth)
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    dunn_index,
    inertia,
    
    # External metrics (with ground truth)
    contingency_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)

from .validation import (
    validate_data,
    validate_labels,
    validate_sample_weight,
    check_n_clusters,
    check_random_state,
    scale_data,
    apply_scaling,
    remove_constant_features,
    handle_missing_values,
    check_array_consistency,
    validate_clustering_input,
    validate_init_params,
    split_data
)

from .device import (
    get_default_device,
    parse_device,
    get_device_info,
    get_optimal_device,
    move_to_device,
    synchronize_device,
    clear_cache,
    estimate_memory_usage,
    check_memory_availability,
    DeviceContext,
    auto_select_device,
    get_batch_size
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
    'MaxIterations',
    
    # Metrics
    'pairwise_distances',
    'silhouette_score',
    'davies_bouldin_score',
    'calinski_harabasz_score',
    'dunn_index',
    'inertia',
    'contingency_matrix',
    'adjusted_rand_score',
    'normalized_mutual_info_score',
    'v_measure_score',
    
    # Validation
    'validate_data',
    'validate_labels',
    'validate_sample_weight',
    'check_n_clusters',
    'check_random_state',
    'scale_data',
    'apply_scaling',
    'remove_constant_features',
    'handle_missing_values',
    'check_array_consistency',
    'validate_clustering_input',
    'validate_init_params',
    'split_data',
    
    # Device management
    'get_default_device',
    'parse_device',
    'get_device_info',
    'get_optimal_device',
    'move_to_device',
    'synchronize_device',
    'clear_cache',
    'estimate_memory_usage',
    'check_memory_availability',
    'DeviceContext',
    'auto_select_device',
    'get_batch_size'
]