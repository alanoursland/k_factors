"""
Device management utilities for GPU/CPU computation.

Provides smart device selection, memory management, and utilities for
moving data between devices efficiently.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import torch
from torch import Tensor
import warnings
import os


def get_default_device() -> torch.device:
    """Get the default device based on availability.
    
    Returns:
        Default device (cuda if available, else cpu)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def parse_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Parse device specification.
    
    Args:
        device: Device specification
            - None: Use default
            - 'auto': Use best available
            - 'cpu': Use CPU
            - 'cuda': Use default CUDA device
            - 'cuda:X': Use CUDA device X
            - 'mps': Use Apple Metal Performance Shaders
            - torch.device: Use as-is
            
    Returns:
        Parsed device
    """
    if device is None or device == 'auto':
        return get_default_device()
        
    if isinstance(device, torch.device):
        return device
        
    if isinstance(device, str):
        if device == 'cpu':
            return torch.device('cpu')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device(device)
        elif device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                warnings.warn("MPS not available, falling back to CPU")
                return torch.device('cpu')
        else:
            raise ValueError(f"Unknown device: {device}")
    else:
        raise TypeError(f"Device must be str or torch.device, got {type(device)}")


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get information about a device.
    
    Args:
        device: Device to query (None for current default)
        
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_default_device()
    else:
        device = parse_device(device)
        
    info = {
        'device': str(device),
        'type': device.type,
        'index': device.index
    }
    
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info.update({
            'name': props.name,
            'total_memory': props.total_memory,
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
            'is_available': torch.cuda.is_available()
        })
        
        # Current memory usage
        info['allocated_memory'] = torch.cuda.memory_allocated(device)
        info['reserved_memory'] = torch.cuda.memory_reserved(device)
        info['free_memory'] = info['total_memory'] - info['allocated_memory']
        
    elif device.type == 'mps':
        info['is_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
    return info


def get_optimal_device(min_memory: Optional[int] = None,
                      prefer_gpu: bool = True) -> torch.device:
    """Get optimal device based on requirements.
    
    Args:
        min_memory: Minimum required memory in bytes
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        Optimal device
    """
    if not prefer_gpu:
        return torch.device('cpu')
        
    # Try CUDA devices
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        
        best_device = None
        best_memory = 0
        
        for i in range(n_devices):
            device = torch.device(f'cuda:{i}')
            info = get_device_info(device)
            free_memory = info['free_memory']
            
            if min_memory is None or free_memory >= min_memory:
                if free_memory > best_memory:
                    best_device = device
                    best_memory = free_memory
                    
        if best_device is not None:
            return best_device
            
    # Try MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Can't query MPS memory, assume it's sufficient
        if min_memory is None:
            return torch.device('mps')
            
    # Fallback to CPU
    return torch.device('cpu')


def move_to_device(data: Any, device: torch.device, 
                  non_blocking: bool = False) -> Any:
    """Recursively move data to device.
    
    Args:
        data: Data to move (tensor, list, dict, etc.)
        device: Target device
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Data on target device
    """
    if isinstance(data, Tensor):
        return data.to(device, non_blocking=non_blocking)
        
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in data]
        return type(data)(moved)
        
    elif isinstance(data, dict):
        return {key: move_to_device(value, device, non_blocking) 
                for key, value in data.items()}
                
    else:
        # Non-tensor data remains unchanged
        return data


def synchronize_device(device: Optional[torch.device] = None) -> None:
    """Synchronize device operations.
    
    Args:
        device: Device to synchronize (None for current)
    """
    if device is None:
        device = get_default_device()
    else:
        device = parse_device(device)
        
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def clear_cache(device: Optional[torch.device] = None) -> None:
    """Clear device memory cache.
    
    Args:
        device: Device to clear (None for all)
    """
    if device is None or device.type == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def estimate_memory_usage(n_samples: int, n_features: int, n_clusters: int,
                         algorithm: str = 'kmeans',
                         dtype: torch.dtype = torch.float32) -> Dict[str, int]:
    """Estimate memory usage for clustering algorithm.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features  
        n_clusters: Number of clusters
        algorithm: Algorithm name
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Bytes per element
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float64:
        bytes_per_element = 8
    elif dtype == torch.float16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4
        
    estimates = {}
    
    # Data matrix
    estimates['data'] = n_samples * n_features * bytes_per_element
    
    # Algorithm-specific
    if algorithm == 'kmeans':
        # Centers
        estimates['centers'] = n_clusters * n_features * bytes_per_element
        # Distances
        estimates['distances'] = n_samples * n_clusters * bytes_per_element
        # Labels
        estimates['labels'] = n_samples * 4  # int32
        
    elif algorithm == 'gmm':
        # Means
        estimates['means'] = n_clusters * n_features * bytes_per_element
        # Covariances (full)
        estimates['covariances'] = n_clusters * n_features * n_features * bytes_per_element
        # Responsibilities
        estimates['responsibilities'] = n_samples * n_clusters * bytes_per_element
        
    elif algorithm in ['kfactors', 'cfactors']:
        # Means
        estimates['means'] = n_clusters * n_features * bytes_per_element
        # Factor loadings (assuming latent_dim ~ sqrt(n_features))
        latent_dim = int(n_features ** 0.5)
        estimates['factors'] = n_clusters * n_features * latent_dim * bytes_per_element
        # Responsibilities/assignments
        estimates['assignments'] = n_samples * n_clusters * bytes_per_element
        
    # Total
    estimates['total'] = sum(estimates.values())
    
    return estimates


def check_memory_availability(required_memory: int, 
                            device: Optional[torch.device] = None,
                            safety_factor: float = 1.2) -> bool:
    """Check if device has enough memory.
    
    Args:
        required_memory: Required memory in bytes
        device: Device to check
        safety_factor: Safety margin multiplier
        
    Returns:
        True if sufficient memory available
    """
    if device is None:
        device = get_default_device()
    else:
        device = parse_device(device)
        
    required_with_safety = int(required_memory * safety_factor)
    
    if device.type == 'cuda':
        info = get_device_info(device)
        return info['free_memory'] >= required_with_safety
    else:
        # Can't easily check CPU/MPS memory, assume it's sufficient
        return True


class DeviceContext:
    """Context manager for temporary device switching.
    
    Example:
        >>> with DeviceContext('cuda:0'):
        ...     # Operations here run on cuda:0
        ...     x = torch.randn(100, 100)
    """
    
    def __init__(self, device: Union[str, torch.device]):
        """
        Args:
            device: Target device
        """
        self.device = parse_device(device)
        self.prev_device = None
        
    def __enter__(self):
        """Enter context."""
        # PyTorch doesn't have a global default device setting,
        # so we'll just return the device for manual use
        return self.device
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Clear cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def auto_select_device(data_size: int, algorithm: str = 'kmeans',
                      min_gpu_memory: int = 1024**3) -> torch.device:
    """Automatically select best device based on data size and algorithm.
    
    Args:
        data_size: Size of data in bytes
        algorithm: Algorithm to use
        min_gpu_memory: Minimum GPU memory to consider GPU
        
    Returns:
        Selected device
    """
    # Small data - use CPU
    if data_size < 10 * 1024**2:  # 10MB
        return torch.device('cpu')
        
    # Check GPU availability and memory
    if torch.cuda.is_available():
        device = get_optimal_device(min_memory=min_gpu_memory)
        if device.type == 'cuda':
            # Estimate total memory usage
            safety_factor = 2.0 if algorithm in ['gmm', 'cfactors'] else 1.5
            if check_memory_availability(data_size, device, safety_factor):
                return device
                
    # Fallback to CPU
    return torch.device('cpu')


def get_batch_size(n_samples: int, n_features: int,
                  device: torch.device, 
                  target_memory_mb: float = 1024) -> int:
    """Calculate appropriate batch size for device memory.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        device: Target device
        target_memory_mb: Target memory usage in MB
        
    Returns:
        Recommended batch size
    """
    bytes_per_sample = n_features * 4  # float32
    target_bytes = target_memory_mb * 1024 * 1024
    
    # For GPU, be more conservative
    if device.type == 'cuda':
        info = get_device_info(device)
        available = info['free_memory'] * 0.8  # Leave 20% buffer
        target_bytes = min(target_bytes, available)
        
    batch_size = int(target_bytes / bytes_per_sample)
    batch_size = max(1, min(batch_size, n_samples))
    
    # Round to nice number
    if batch_size > 1000:
        batch_size = (batch_size // 1000) * 1000
    elif batch_size > 100:
        batch_size = (batch_size // 100) * 100
        
    return batch_size