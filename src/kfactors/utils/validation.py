"""
Input validation and preprocessing utilities.

Provides functions for validating and preprocessing data before clustering,
including handling of edge cases, data type conversion, and sanity checks.
"""

from typing import Optional, Union, Tuple, List
import torch
from torch import Tensor
import numpy as np
import warnings


def validate_data(X: Union[Tensor, np.ndarray, list], 
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 ensure_2d: bool = True,
                 ensure_finite: bool = True,
                 ensure_min_samples: int = 1,
                 ensure_min_features: int = 1,
                 copy: bool = False) -> Tensor:
    """Validate and convert input data to tensor.
    
    Args:
        X: Input data (tensor, numpy array, or list)
        dtype: Target data type
        device: Target device
        ensure_2d: Whether to ensure 2D shape
        ensure_finite: Whether to check for inf/nan
        ensure_min_samples: Minimum number of samples required
        ensure_min_features: Minimum number of features required  
        copy: Whether to force a copy
        
    Returns:
        Validated tensor
        
    Raises:
        ValueError: If validation fails
    """
    # Convert to tensor
    if isinstance(X, Tensor):
        if copy or X.dtype != dtype or (device is not None and X.device != device):
            X = X.to(dtype=dtype, device=device)
    elif isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(dtype=dtype, device=device)
    elif isinstance(X, list):
        X = torch.tensor(X, dtype=dtype, device=device)
    else:
        raise TypeError(f"Cannot convert {type(X)} to tensor")
        
    # Ensure 2D
    if ensure_2d:
        if X.dim() == 1:
            X = X.unsqueeze(1)
        elif X.dim() != 2:
            raise ValueError(f"Expected 2D array, got {X.dim()}D")
            
    # Check shape
    if ensure_2d:
        n_samples, n_features = X.shape
        
        if n_samples < ensure_min_samples:
            raise ValueError(f"Found {n_samples} samples, but need at least "
                           f"{ensure_min_samples}")
                           
        if n_features < ensure_min_features:
            raise ValueError(f"Found {n_features} features, but need at least "
                           f"{ensure_min_features}")
                           
    # Check for finite values
    if ensure_finite:
        if torch.isnan(X).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(X).any():
            raise ValueError("Input contains infinite values")
            
    return X


def validate_labels(labels: Union[Tensor, np.ndarray, list],
                   n_samples: Optional[int] = None,
                   ensure_consecutive: bool = False) -> Tensor:
    """Validate cluster labels.
    
    Args:
        labels: Cluster labels
        n_samples: Expected number of samples
        ensure_consecutive: Whether labels should be 0, 1, ..., k-1
        
    Returns:
        Validated label tensor
        
    Raises:
        ValueError: If validation fails
    """
    # Convert to tensor
    if isinstance(labels, Tensor):
        labels = labels.long()
    elif isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    elif isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        raise TypeError(f"Cannot convert {type(labels)} to label tensor")
        
    # Check dimension
    if labels.dim() != 1:
        raise ValueError(f"Labels must be 1D, got {labels.dim()}D")
        
    # Check length
    if n_samples is not None and len(labels) != n_samples:
        raise ValueError(f"Expected {n_samples} labels, got {len(labels)}")
        
    # Check values
    if (labels < 0).any():
        raise ValueError("Labels must be non-negative")
        
    # Check consecutive
    if ensure_consecutive:
        unique_labels = torch.unique(labels)
        expected = torch.arange(len(unique_labels), device=labels.device)
        if not torch.equal(unique_labels, expected):
            warnings.warn("Labels are not consecutive integers starting from 0")
            
    return labels


def validate_sample_weight(sample_weight: Optional[Union[Tensor, np.ndarray]],
                          n_samples: int) -> Optional[Tensor]:
    """Validate sample weights.
    
    Args:
        sample_weight: Sample weights or None
        n_samples: Number of samples
        
    Returns:
        Validated weight tensor or None
    """
    if sample_weight is None:
        return None
        
    # Convert to tensor
    if isinstance(sample_weight, np.ndarray):
        sample_weight = torch.from_numpy(sample_weight).float()
    elif not isinstance(sample_weight, Tensor):
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        
    # Check shape
    if sample_weight.dim() != 1:
        raise ValueError(f"Sample weights must be 1D, got {sample_weight.dim()}D")
        
    if len(sample_weight) != n_samples:
        raise ValueError(f"Expected {n_samples} weights, got {len(sample_weight)}")
        
    # Check values
    if (sample_weight < 0).any():
        raise ValueError("Sample weights must be non-negative")
        
    if sample_weight.sum() == 0:
        raise ValueError("Sample weights sum to zero")
        
    return sample_weight


def check_n_clusters(n_clusters: int, n_samples: int) -> None:
    """Validate number of clusters.
    
    Args:
        n_clusters: Number of clusters
        n_samples: Number of samples
        
    Raises:
        ValueError: If invalid
    """
    if not isinstance(n_clusters, int):
        raise TypeError(f"n_clusters must be int, got {type(n_clusters)}")
        
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) cannot be larger than "
                        f"n_samples ({n_samples})")
                        

def check_random_state(random_state: Optional[Union[int, torch.Generator]]) -> Optional[torch.Generator]:
    """Create generator from random state.
    
    Args:
        random_state: Seed or generator
        
    Returns:
        Generator or None
    """
    if random_state is None:
        return None
    elif isinstance(random_state, int):
        generator = torch.Generator()
        generator.manual_seed(random_state)
        return generator
    elif isinstance(random_state, torch.Generator):
        return random_state
    else:
        raise TypeError(f"random_state must be int or Generator, got {type(random_state)}")


def scale_data(X: Tensor, method: str = 'standard',
               return_params: bool = False) -> Union[Tensor, Tuple[Tensor, dict]]:
    """Scale/normalize data.
    
    Args:
        X: (n, d) data tensor
        method: Scaling method
            - 'standard': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'maxabs': Scale to [-1, 1]
            - 'robust': Median and MAD
            - 'normalize': Unit norm rows
        return_params: Whether to return scaling parameters
        
    Returns:
        Scaled data and optionally scaling parameters
    """
    if method == 'standard':
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        X_scaled = (X - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_vals = X.min(dim=0, keepdim=True)[0]
        max_vals = X.max(dim=0, keepdim=True)[0]
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        X_scaled = (X - min_vals) / range_vals
        params = {'min': min_vals, 'range': range_vals}
        
    elif method == 'maxabs':
        max_abs = X.abs().max(dim=0, keepdim=True)[0]
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        X_scaled = X / max_abs
        params = {'max_abs': max_abs}
        
    elif method == 'robust':
        median = X.median(dim=0, keepdim=True)[0]
        mad = (X - median).abs().median(dim=0, keepdim=True)[0]
        mad = torch.where(mad == 0, torch.ones_like(mad), mad)
        X_scaled = (X - median) / (1.4826 * mad)  # Consistent estimator
        params = {'median': median, 'mad': mad}
        
    elif method == 'normalize':
        norms = X.norm(dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        X_scaled = X / norms
        params = {'norms': norms}
        
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    if return_params:
        return X_scaled, params
    else:
        return X_scaled


def apply_scaling(X: Tensor, params: dict, method: str) -> Tensor:
    """Apply scaling parameters to new data.
    
    Args:
        X: Data to scale
        params: Scaling parameters from scale_data
        method: Scaling method used
        
    Returns:
        Scaled data
    """
    if method == 'standard':
        return (X - params['mean']) / params['std']
    elif method == 'minmax':
        return (X - params['min']) / params['range']
    elif method == 'maxabs':
        return X / params['max_abs']
    elif method == 'robust':
        return (X - params['median']) / (1.4826 * params['mad'])
    elif method == 'normalize':
        norms = X.norm(dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        return X / norms
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def remove_constant_features(X: Tensor, tol: float = 1e-10) -> Tuple[Tensor, Tensor]:
    """Remove features with zero variance.
    
    Args:
        X: (n, d) data
        tol: Tolerance for zero variance
        
    Returns:
        X with constant features removed
        Mask of kept features
    """
    variances = X.var(dim=0)
    mask = variances > tol
    
    if mask.sum() == 0:
        raise ValueError("All features have zero variance")
        
    if mask.sum() < len(mask):
        warnings.warn(f"Removed {len(mask) - mask.sum()} constant features")
        
    return X[:, mask], mask


def handle_missing_values(X: Tensor, method: str = 'error') -> Tensor:
    """Handle missing values in data.
    
    Args:
        X: Data potentially containing NaN
        method: How to handle missing values
            - 'error': Raise error if NaN found
            - 'drop_samples': Remove samples with NaN
            - 'drop_features': Remove features with NaN
            - 'mean': Impute with feature means
            - 'median': Impute with feature medians
            - 'zero': Replace with zeros
            
    Returns:
        Data with missing values handled
    """
    nan_mask = torch.isnan(X)
    
    if not nan_mask.any():
        return X
        
    if method == 'error':
        raise ValueError("Data contains NaN values")
        
    elif method == 'drop_samples':
        keep_samples = ~nan_mask.any(dim=1)
        if keep_samples.sum() == 0:
            raise ValueError("All samples contain NaN")
        return X[keep_samples]
        
    elif method == 'drop_features':
        keep_features = ~nan_mask.any(dim=0)
        if keep_features.sum() == 0:
            raise ValueError("All features contain NaN")
        return X[:, keep_features]
        
    elif method == 'mean':
        X = X.clone()
        for j in range(X.shape[1]):
            col = X[:, j]
            if nan_mask[:, j].any():
                mean_val = col[~nan_mask[:, j]].mean()
                X[nan_mask[:, j], j] = mean_val
        return X
        
    elif method == 'median':
        X = X.clone()
        for j in range(X.shape[1]):
            col = X[:, j]
            if nan_mask[:, j].any():
                median_val = col[~nan_mask[:, j]].median()
                X[nan_mask[:, j], j] = median_val
        return X
        
    elif method == 'zero':
        X = X.clone()
        X[nan_mask] = 0
        return X
        
    else:
        raise ValueError(f"Unknown method: {method}")


def check_array_consistency(*arrays: Tensor) -> None:
    """Check that arrays have consistent shapes.
    
    Args:
        *arrays: Tensors to check
        
    Raises:
        ValueError: If inconsistent
    """
    if len(arrays) < 2:
        return
        
    n_samples = None
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
            
        if n_samples is None:
            n_samples = len(arr)
        elif len(arr) != n_samples:
            raise ValueError(f"Array {i} has {len(arr)} samples, "
                           f"expected {n_samples}")


def validate_clustering_input(X: Union[Tensor, np.ndarray],
                            n_clusters: int,
                            sample_weight: Optional[Union[Tensor, np.ndarray]] = None,
                            random_state: Optional[Union[int, torch.Generator]] = None,
                            dtype: torch.dtype = torch.float32,
                            device: Optional[torch.device] = None) -> dict:
    """Comprehensive validation for clustering input.
    
    Args:
        X: Input data
        n_clusters: Number of clusters
        sample_weight: Optional sample weights
        random_state: Random state
        dtype: Data type
        device: Device
        
    Returns:
        Dictionary with validated inputs
    """
    # Validate data
    X = validate_data(X, dtype=dtype, device=device, ensure_2d=True,
                     ensure_finite=True, ensure_min_samples=n_clusters)
    
    n_samples, n_features = X.shape
    
    # Validate n_clusters
    check_n_clusters(n_clusters, n_samples)
    
    # Validate sample weights
    sample_weight = validate_sample_weight(sample_weight, n_samples)
    
    # Validate random state
    generator = check_random_state(random_state)
    
    return {
        'X': X,
        'n_clusters': n_clusters,
        'n_samples': n_samples,
        'n_features': n_features,
        'sample_weight': sample_weight,
        'generator': generator
    }


def validate_init_params(init: Union[str, Tensor, List[Tensor]],
                        n_clusters: int,
                        n_features: int) -> Union[str, Tensor]:
    """Validate initialization parameters.
    
    Args:
        init: Initialization method or initial centers
        n_clusters: Number of clusters
        n_features: Number of features
        
    Returns:
        Validated initialization
    """
    if isinstance(init, str):
        valid_methods = ['k-means++', 'random', 'kmeans']
        if init not in valid_methods:
            raise ValueError(f"init must be one of {valid_methods}, got '{init}'")
        return init
        
    elif isinstance(init, (Tensor, np.ndarray)):
        init_tensor = validate_data(init, ensure_2d=True)
        
        if init_tensor.shape != (n_clusters, n_features):
            raise ValueError(f"init array must have shape ({n_clusters}, {n_features}), "
                           f"got {init_tensor.shape}")
        return init_tensor
        
    elif isinstance(init, list):
        if len(init) != n_clusters:
            raise ValueError(f"init list must have {n_clusters} elements, "
                           f"got {len(init)}")
                           
        init_tensors = []
        for i, center in enumerate(init):
            center_tensor = validate_data(center, ensure_2d=False)
            if center_tensor.shape != (n_features,):
                raise ValueError(f"init[{i}] must have shape ({n_features},), "
                               f"got {center_tensor.shape}")
            init_tensors.append(center_tensor)
            
        return torch.stack(init_tensors)
        
    else:
        raise TypeError(f"init must be str, array, or list, got {type(init)}")


def split_data(X: Tensor, test_size: Union[float, int] = 0.2,
               random_state: Optional[int] = None,
               stratify: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split data into train and test sets.
    
    Args:
        X: Data to split
        test_size: Fraction or number of test samples
        random_state: Random seed
        stratify: Labels for stratified split
        
    Returns:
        X_train, X_test, indices_train, indices_test
    """
    n_samples = len(X)
    
    # Determine test size
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
        
    n_train = n_samples - n_test
    
    # Create indices
    indices = torch.arange(n_samples, device=X.device)
    
    if random_state is not None:
        generator = torch.Generator(device=X.device)
        generator.manual_seed(random_state)
    else:
        generator = None
        
    if stratify is not None:
        # Stratified split
        train_indices = []
        test_indices = []
        
        unique_labels = torch.unique(stratify)
        for label in unique_labels:
            label_indices = indices[stratify == label]
            n_label = len(label_indices)
            n_test_label = max(1, int(n_label * test_size))
            
            perm = torch.randperm(n_label, generator=generator, device=X.device)
            test_indices.append(label_indices[perm[:n_test_label]])
            train_indices.append(label_indices[perm[n_test_label:]])
            
        train_indices = torch.cat(train_indices)
        test_indices = torch.cat(test_indices)
        
    else:
        # Random split
        perm = torch.randperm(n_samples, generator=generator, device=X.device)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
        
    return X[train_indices], X[test_indices], train_indices, test_indices