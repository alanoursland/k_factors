"""
Clustering evaluation metrics.

Provides various metrics for evaluating clustering quality, including
internal metrics (no ground truth needed) and external metrics (comparing
to ground truth labels).
"""

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import warnings


def pairwise_distances(X: Tensor, Y: Optional[Tensor] = None, 
                      metric: str = 'euclidean') -> Tensor:
    """Compute pairwise distances between points.
    
    Args:
        X: (n, d) first set of points
        Y: (m, d) second set of points (if None, uses X)
        metric: Distance metric ('euclidean', 'cosine')
        
    Returns:
        (n, m) distance matrix
    """
    if Y is None:
        Y = X
        
    if metric == 'euclidean':
        # Efficient computation using broadcasting
        # ||x - y||² = ||x||² + ||y||² - 2<x,y>
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
        XY = torch.matmul(X, Y.t())
        distances = X_norm + Y_norm.t() - 2 * XY
        distances = torch.clamp(distances, min=0.0)  # Numerical safety
        return torch.sqrt(distances)
        
    elif metric == 'cosine':
        # Normalize rows
        X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-8)
        Y_norm = Y / torch.norm(Y, dim=1, keepdim=True).clamp(min=1e-8)
        # Cosine similarity
        similarities = torch.matmul(X_norm, Y_norm.t())
        # Convert to distance
        return 1 - similarities
        
    else:
        raise ValueError(f"Unknown metric: {metric}")


def silhouette_score(X: Tensor, labels: Tensor, metric: str = 'euclidean',
                    sample_size: Optional[int] = None) -> float:
    """Compute mean Silhouette Coefficient.
    
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).
    
    Args:
        X: (n, d) data points
        labels: (n,) cluster labels
        metric: Distance metric
        sample_size: If provided, subsample for efficiency
        
    Returns:
        Mean silhouette coefficient in [-1, 1]
    """
    n_samples = len(X)
    n_clusters = labels.max().item() + 1
    
    if n_clusters == 1:
        return 0.0
        
    # Subsample if requested
    if sample_size is not None and sample_size < n_samples:
        indices = torch.randperm(n_samples)[:sample_size]
        X = X[indices]
        labels = labels[indices]
        n_samples = sample_size
        
    # Compute pairwise distances
    distances = pairwise_distances(X, metric=metric)
    
    silhouette_values = torch.zeros(n_samples, device=X.device)
    
    for i in range(n_samples):
        # Find points in same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        
        if same_cluster.sum() == 0:
            silhouette_values[i] = 0
            continue
            
        # Mean intra-cluster distance
        a = distances[i, same_cluster].mean()
        
        # Mean distance to other clusters
        b_values = []
        for k in range(n_clusters):
            if k == labels[i]:
                continue
            other_cluster = labels == k
            if other_cluster.sum() > 0:
                b_values.append(distances[i, other_cluster].mean())
                
        if len(b_values) == 0:
            silhouette_values[i] = 0
            continue
            
        b = torch.stack(b_values).min()
        
        # Silhouette coefficient
        silhouette_values[i] = (b - a) / torch.max(a, b)
        
    return silhouette_values.mean().item()


def davies_bouldin_score(X: Tensor, labels: Tensor, centers: Optional[Tensor] = None) -> float:
    """Compute Davies-Bouldin score.
    
    Lower values indicate better clustering. The score is defined as the
    average similarity measure of each cluster with its most similar cluster.
    
    Args:
        X: (n, d) data points
        labels: (n,) cluster labels
        centers: (k, d) cluster centers (computed if not provided)
        
    Returns:
        Davies-Bouldin score (lower is better)
    """
    n_clusters = labels.max().item() + 1
    device = X.device
    
    if n_clusters == 1:
        return 0.0
        
    # Compute cluster centers if not provided
    if centers is None:
        centers = torch.zeros(n_clusters, X.shape[1], device=device)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(dim=0)
                
    # Compute within-cluster scatter
    scatter = torch.zeros(n_clusters, device=device)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_points = X[mask]
            distances = torch.norm(cluster_points - centers[k], dim=1)
            scatter[k] = distances.mean()
            
    # Compute between-cluster distances
    center_distances = pairwise_distances(centers)
    
    # Compute Davies-Bouldin index
    db_values = torch.zeros(n_clusters, device=device)
    for i in range(n_clusters):
        ratios = []
        for j in range(n_clusters):
            if i != j and center_distances[i, j] > 0:
                ratio = (scatter[i] + scatter[j]) / center_distances[i, j]
                ratios.append(ratio)
                
        if len(ratios) > 0:
            db_values[i] = torch.stack(ratios).max()
            
    return db_values.mean().item()


def calinski_harabasz_score(X: Tensor, labels: Tensor) -> float:
    """Compute Calinski-Harabasz score (Variance Ratio Criterion).
    
    Higher values indicate better defined clusters.
    
    Args:
        X: (n, d) data points
        labels: (n,) cluster labels
        
    Returns:
        Calinski-Harabasz score (higher is better)
    """
    n_samples = len(X)
    n_clusters = labels.max().item() + 1
    
    if n_clusters == 1:
        return 1.0
        
    # Overall mean
    mean = X.mean(dim=0)
    
    # Between-group sum of squares
    bgss = torch.tensor(0.0, device=X.device)
    # Within-group sum of squares
    wgss = torch.tensor(0.0, device=X.device)
    
    for k in range(n_clusters):
        mask = labels == k
        n_k = mask.sum()
        
        if n_k > 0:
            cluster_points = X[mask]
            cluster_mean = cluster_points.mean(dim=0)
            
            # Between-group dispersion
            bgss += n_k * torch.sum((cluster_mean - mean) ** 2)
            
            # Within-group dispersion
            wgss += torch.sum((cluster_points - cluster_mean) ** 2)
            
    # Calinski-Harabasz score
    if wgss == 0:
        return 0.0
        
    ch_score = (bgss / wgss) * ((n_samples - n_clusters) / (n_clusters - 1))
    return ch_score.item()


def dunn_index(X: Tensor, labels: Tensor, metric: str = 'euclidean') -> float:
    """Compute Dunn index.
    
    Higher values indicate better clustering (compact, well-separated clusters).
    
    Args:
        X: (n, d) data points
        labels: (n,) cluster labels
        metric: Distance metric
        
    Returns:
        Dunn index (higher is better)
    """
    n_clusters = labels.max().item() + 1
    
    if n_clusters == 1:
        return float('inf')
        
    # Compute pairwise distances
    distances = pairwise_distances(X, metric=metric)
    
    # Minimum inter-cluster distance
    min_inter_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mask_i = labels == i
            mask_j = labels == j
            
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                inter_dists = distances[mask_i][:, mask_j]
                min_inter_dist = min(min_inter_dist, inter_dists.min().item())
                
    # Maximum intra-cluster diameter
    max_intra_diam = 0.0
    for k in range(n_clusters):
        mask = labels == k
        n_k = mask.sum()
        
        if n_k > 1:
            cluster_dists = distances[mask][:, mask]
            max_intra_diam = max(max_intra_diam, cluster_dists.max().item())
            
    if max_intra_diam == 0:
        return float('inf')
        
    return min_inter_dist / max_intra_diam


def inertia(X: Tensor, labels: Tensor, centers: Tensor) -> float:
    """Compute sum of squared distances to nearest centers (inertia).
    
    Args:
        X: (n, d) data points
        labels: (n,) cluster labels
        centers: (k, d) cluster centers
        
    Returns:
        Total inertia (lower is better)
    """
    total = 0.0
    n_clusters = centers.shape[0]
    
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_points = X[mask]
            distances = torch.sum((cluster_points - centers[k]) ** 2, dim=1)
            total += distances.sum().item()
            
    return total


# External metrics (require ground truth)

def contingency_matrix(labels_true: Tensor, labels_pred: Tensor) -> Tensor:
    """Build contingency matrix for comparing clusterings.
    
    Args:
        labels_true: (n,) true labels
        labels_pred: (n,) predicted labels
        
    Returns:
        Contingency matrix C where C[i,j] is the number of samples
        with true label i and predicted label j
    """
    n_true = labels_true.max().item() + 1
    n_pred = labels_pred.max().item() + 1
    
    matrix = torch.zeros(n_true, n_pred, dtype=torch.long, 
                        device=labels_true.device)
    
    for i in range(len(labels_true)):
        matrix[labels_true[i], labels_pred[i]] += 1
        
    return matrix


def adjusted_rand_score(labels_true: Tensor, labels_pred: Tensor) -> float:
    """Compute Adjusted Rand Index.
    
    ARI is 1.0 for perfect match, 0.0 for random labeling.
    
    Args:
        labels_true: (n,) ground truth labels
        labels_pred: (n,) predicted labels
        
    Returns:
        ARI score in [-1, 1]
    """
    # Contingency matrix
    contingency = contingency_matrix(labels_true, labels_pred)
    
    # Sum over rows and columns
    row_sum = contingency.sum(dim=1)
    col_sum = contingency.sum(dim=0)
    n = contingency.sum()
    
    # Rand index
    sum_comb_c = torch.sum(contingency * (contingency - 1)) / 2
    sum_comb_r = torch.sum(row_sum * (row_sum - 1)) / 2
    sum_comb_c = torch.sum(col_sum * (col_sum - 1)) / 2
    
    # Expected index
    expected_index = sum_comb_r * sum_comb_c / (n * (n - 1) / 2)
    max_index = (sum_comb_r + sum_comb_c) / 2
    
    if max_index - expected_index == 0:
        return 1.0
        
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari.item()


def normalized_mutual_info_score(labels_true: Tensor, labels_pred: Tensor,
                               average_method: str = 'arithmetic') -> float:
    """Compute Normalized Mutual Information.
    
    NMI is 1.0 for perfect match, 0.0 for independent labelings.
    
    Args:
        labels_true: (n,) ground truth labels
        labels_pred: (n,) predicted labels
        average_method: How to average ('arithmetic', 'geometric', 'max', 'min')
        
    Returns:
        NMI score in [0, 1]
    """
    contingency = contingency_matrix(labels_true, labels_pred).float()
    
    # Marginals
    row_sum = contingency.sum(dim=1)
    col_sum = contingency.sum(dim=0)
    n = contingency.sum()
    
    # Entropy calculations
    def entropy(counts):
        p = counts / counts.sum()
        p = p[p > 0]  # Remove zeros
        return -torch.sum(p * torch.log(p))
        
    h_true = entropy(row_sum)
    h_pred = entropy(col_sum)
    
    # Mutual information
    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i, j] > 0:
                mi += (contingency[i, j] / n) * torch.log(
                    contingency[i, j] * n / (row_sum[i] * col_sum[j])
                )
                
    # Normalize
    if average_method == 'arithmetic':
        denominator = (h_true + h_pred) / 2
    elif average_method == 'geometric':
        denominator = torch.sqrt(h_true * h_pred)
    elif average_method == 'max':
        denominator = torch.max(h_true, h_pred)
    elif average_method == 'min':
        denominator = torch.min(h_true, h_pred)
    else:
        raise ValueError(f"Unknown average method: {average_method}")
        
    if denominator == 0:
        return 1.0 if mi == 0 else 0.0
        
    return (mi / denominator).item()


def v_measure_score(labels_true: Tensor, labels_pred: Tensor, beta: float = 1.0) -> Tuple[float, float, float]:
    """Compute V-measure (homogeneity, completeness, v-measure).
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        beta: Weight for harmonic mean
        
    Returns:
        homogeneity: Each cluster contains only one class
        completeness: All points of a class are in one cluster
        v_measure: Harmonic mean of homogeneity and completeness
    """
    contingency = contingency_matrix(labels_true, labels_pred).float()
    n = contingency.sum()
    
    # Marginals
    row_sum = contingency.sum(dim=1)
    col_sum = contingency.sum(dim=0)
    
    # Conditional entropies
    h_c_given_k = 0.0  # H(C|K)
    h_k_given_c = 0.0  # H(K|C)
    
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i, j] > 0:
                h_c_given_k -= (contingency[i, j] / n) * torch.log(
                    contingency[i, j] / col_sum[j]
                )
                h_k_given_c -= (contingency[i, j] / n) * torch.log(
                    contingency[i, j] / row_sum[i]
                )
                
    # Entropies
    def entropy(counts):
        p = counts / n
        p = p[p > 0]
        return -torch.sum(p * torch.log(p))
        
    h_c = entropy(row_sum)
    h_k = entropy(col_sum)
    
    # Homogeneity
    homogeneity = 1.0 if h_c == 0 else 1.0 - h_c_given_k / h_c
    
    # Completeness
    completeness = 1.0 if h_k == 0 else 1.0 - h_k_given_c / h_k
    
    # V-measure
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = ((1 + beta) * homogeneity * completeness / 
                     (beta * homogeneity + completeness))
        
    return homogeneity.item(), completeness.item(), v_measure.item()