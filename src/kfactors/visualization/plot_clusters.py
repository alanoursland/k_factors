"""
Cluster visualization utilities.

Provides functions for visualizing clustering results in 2D and 3D,
including cluster boundaries, centers, and membership strengths.
"""

from typing import Optional, Union, Tuple, List, Dict, Any
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import warnings


def plot_clusters_2d(X: Tensor, 
                    labels: Tensor,
                    centers: Optional[Tensor] = None,
                    ax: Optional[plt.Axes] = None,
                    colors: Optional[List[str]] = None,
                    markers: Optional[List[str]] = None,
                    alpha: float = 0.7,
                    center_marker: str = 'X',
                    center_size: int = 200,
                    point_size: int = 50,
                    show_legend: bool = True,
                    title: Optional[str] = None) -> plt.Axes:
    """Plot 2D clustering results.
    
    Args:
        X: (n, 2) data points
        labels: (n,) cluster labels
        centers: Optional (k, 2) cluster centers
        ax: Matplotlib axes (created if None)
        colors: List of colors for clusters
        markers: List of markers for clusters
        alpha: Point transparency
        center_marker: Marker for centers
        center_size: Size of center markers
        point_size: Size of data points
        show_legend: Whether to show legend
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Convert to numpy for matplotlib
    X_np = X.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get unique labels
    unique_labels = np.unique(labels_np)
    n_clusters = len(unique_labels)
    
    # Default colors
    if colors is None:
        cmap = cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
        
    # Default markers
    if markers is None:
        markers = ['o'] * n_clusters
        
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        ax.scatter(X_np[mask, 0], X_np[mask, 1],
                  c=[colors[i]], 
                  marker=markers[i % len(markers)],
                  s=point_size,
                  alpha=alpha,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f'Cluster {label}')
                  
    # Plot centers
    if centers is not None:
        centers_np = centers.cpu().numpy()
        ax.scatter(centers_np[:, 0], centers_np[:, 1],
                  c='black',
                  marker=center_marker,
                  s=center_size,
                  edgecolors='white',
                  linewidth=2,
                  label='Centers',
                  zorder=10)
                  
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    if title:
        ax.set_title(title)
        
    if show_legend:
        ax.legend()
        
    return ax


def plot_clusters_3d(X: Tensor,
                    labels: Tensor,
                    centers: Optional[Tensor] = None,
                    ax: Optional[Axes3D] = None,
                    colors: Optional[List[str]] = None,
                    alpha: float = 0.7,
                    center_size: int = 200,
                    point_size: int = 50,
                    elev: float = 30,
                    azim: float = 45,
                    title: Optional[str] = None) -> Axes3D:
    """Plot 3D clustering results.
    
    Args:
        X: (n, 3) data points
        labels: (n,) cluster labels
        centers: Optional (k, 3) cluster centers
        ax: 3D axes (created if None)
        colors: List of colors
        alpha: Point transparency
        center_size: Size of center markers
        point_size: Size of data points
        elev: Elevation angle
        azim: Azimuth angle
        title: Plot title
        
    Returns:
        3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
    # Convert to numpy
    X_np = X.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get unique labels
    unique_labels = np.unique(labels_np)
    n_clusters = len(unique_labels)
    
    # Default colors
    if colors is None:
        cmap = cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
        
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        ax.scatter(X_np[mask, 0], X_np[mask, 1], X_np[mask, 2],
                  c=[colors[i]],
                  s=point_size,
                  alpha=alpha,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f'Cluster {label}')
                  
    # Plot centers
    if centers is not None:
        centers_np = centers.cpu().numpy()
        ax.scatter(centers_np[:, 0], centers_np[:, 1], centers_np[:, 2],
                  c='black',
                  marker='X',
                  s=center_size,
                  edgecolors='white',
                  linewidth=2,
                  label='Centers')
                  
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    
    ax.view_init(elev=elev, azim=azim)
    
    if title:
        ax.set_title(title)
        
    ax.legend()
    
    return ax


def plot_fuzzy_clusters(X: Tensor,
                       memberships: Tensor,
                       centers: Optional[Tensor] = None,
                       ax: Optional[plt.Axes] = None,
                       threshold: float = 0.1,
                       show_all: bool = False,
                       cmap: str = 'viridis',
                       title: Optional[str] = None) -> plt.Axes:
    """Plot fuzzy clustering with membership strengths.
    
    Args:
        X: (n, 2) data points
        memberships: (n, k) membership matrix
        centers: Optional (k, 2) cluster centers
        ax: Matplotlib axes
        threshold: Minimum membership to show
        show_all: If True, show all points with size based on membership
        cmap: Colormap name
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    X_np = X.cpu().numpy()
    memberships_np = memberships.cpu().numpy()
    
    n_clusters = memberships.shape[1]
    colormap = cm.get_cmap(cmap)
    
    if show_all:
        # Show all points with color mixing based on memberships
        # Get dominant cluster and its membership
        max_membership, dominant_cluster = memberships.max(dim=1)
        max_membership_np = max_membership.cpu().numpy()
        dominant_cluster_np = dominant_cluster.cpu().numpy()
        
        # Create colors based on membership strength
        colors = []
        for i in range(len(X)):
            # Mix cluster colors based on memberships
            color = np.zeros(3)
            for k in range(n_clusters):
                cluster_color = colormap(k / n_clusters)[:3]
                color += memberships_np[i, k] * np.array(cluster_color)
            colors.append(color)
            
        # Point sizes based on max membership
        sizes = 20 + 80 * max_membership_np
        
        scatter = ax.scatter(X_np[:, 0], X_np[:, 1],
                           c=colors,
                           s=sizes,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
    else:
        # Show only points with membership above threshold
        for k in range(n_clusters):
            mask = memberships_np[:, k] > threshold
            if mask.any():
                # Size proportional to membership
                sizes = 100 * memberships_np[mask, k]
                color = colormap(k / n_clusters)
                
                ax.scatter(X_np[mask, 0], X_np[mask, 1],
                         c=[color],
                         s=sizes,
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=0.5,
                         label=f'Cluster {k}')
                         
    # Plot centers
    if centers is not None:
        centers_np = centers.cpu().numpy()
        ax.scatter(centers_np[:, 0], centers_np[:, 1],
                  c='red',
                  marker='X',
                  s=300,
                  edgecolors='white',
                  linewidth=2,
                  label='Centers',
                  zorder=10)
                  
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Fuzzy Clustering Results')
        
    if not show_all:
        ax.legend()
        
    return ax


def plot_cluster_boundaries(X: Tensor,
                          model: Any,
                          ax: Optional[plt.Axes] = None,
                          resolution: int = 100,
                          alpha: float = 0.3,
                          show_data: bool = True,
                          show_centers: bool = True,
                          title: Optional[str] = None) -> plt.Axes:
    """Plot decision boundaries for clustering.
    
    Args:
        X: (n, 2) data points
        model: Fitted clustering model with predict method
        ax: Matplotlib axes
        resolution: Grid resolution
        alpha: Boundary transparency
        show_data: Whether to show data points
        show_centers: Whether to show centers
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    X_np = X.cpu().numpy()
    
    # Create mesh grid
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
                        
    # Predict on mesh
    mesh_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], 
                              dtype=torch.float32,
                              device=X.device)
    
    Z = model.predict(mesh_points)
    Z_np = Z.cpu().numpy().reshape(xx.shape)
    
    # Plot boundaries
    ax.contourf(xx, yy, Z_np, alpha=alpha, cmap='viridis')
    
    # Plot data points
    if show_data:
        labels = model.predict(X)
        plot_clusters_2d(X, labels, ax=ax, show_legend=False)
        
    # Plot centers
    if show_centers and hasattr(model, 'cluster_centers_'):
        centers = model.cluster_centers_
        centers_np = centers.cpu().numpy()
        ax.scatter(centers_np[:, 0], centers_np[:, 1],
                  c='red',
                  marker='X',
                  s=300,
                  edgecolors='white',
                  linewidth=2,
                  zorder=10)
                  
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Cluster Decision Boundaries')
        
    return ax


def plot_gaussian_ellipses(means: Tensor,
                          covariances: Tensor,
                          ax: Optional[plt.Axes] = None,
                          n_std: float = 2.0,
                          alpha: float = 0.3,
                          colors: Optional[List[str]] = None,
                          show_centers: bool = True,
                          title: Optional[str] = None) -> plt.Axes:
    """Plot Gaussian components as ellipses.
    
    Args:
        means: (k, 2) cluster means
        covariances: (k, 2, 2) covariance matrices
        ax: Matplotlib axes
        n_std: Number of standard deviations
        alpha: Ellipse transparency
        colors: List of colors
        show_centers: Whether to show centers
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    means_np = means.cpu().numpy()
    cov_np = covariances.cpu().numpy()
    
    n_clusters = len(means)
    
    if colors is None:
        cmap = cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
        
    for k in range(n_clusters):
        # Compute ellipse parameters
        mean = means_np[k]
        cov = cov_np[k]
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Ellipse parameters
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * n_std * np.sqrt(eigvals[0])
        height = 2 * n_std * np.sqrt(eigvals[1])
        
        # Create ellipse
        ellipse = Ellipse(mean, width, height,
                         angle=angle,
                         facecolor=colors[k],
                         alpha=alpha,
                         edgecolor=colors[k],
                         linewidth=2,
                         label=f'Cluster {k}')
        ax.add_patch(ellipse)
        
        # Plot center
        if show_centers:
            ax.scatter(mean[0], mean[1],
                      c=[colors[k]],
                      marker='o',
                      s=100,
                      edgecolors='black',
                      linewidth=1,
                      zorder=10)
                      
    ax.set_aspect('equal')
    ax.autoscale()
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Gaussian Components ({n_std}σ ellipses)')
        
    ax.legend()
    
    return ax