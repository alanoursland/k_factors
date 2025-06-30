"""
Demo of K-Factors clustering algorithm.

This example shows how to:
1. Generate synthetic data with local subspace structure
2. Apply K-Factors clustering
3. Visualize results and compare with K-means
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
import sys
sys.path.append('..')

from kfactors.algorithms.kfactors import KFactors
from kfactors.algorithms.kmeans import KMeans


def generate_subspace_data(n_points_per_cluster=200, n_clusters=3, 
                          ambient_dim=10, subspace_dim=2, noise_level=0.1):
    """Generate synthetic data with local subspace structure.
    
    Each cluster lies approximately on a different 2D subspace in 10D space.
    """
    torch.manual_seed(42)
    
    data_list = []
    true_labels = []
    
    for k in range(n_clusters):
        # Random orthonormal basis for this cluster's subspace
        basis = torch.randn(ambient_dim, subspace_dim)
        basis, _ = torch.linalg.qr(basis)
        
        # Random mean for this cluster
        mean = torch.randn(ambient_dim) * 3
        
        # Generate points in subspace
        # Coefficients in subspace (make them somewhat spread out)
        coeffs = torch.randn(n_points_per_cluster, subspace_dim) * 2
        
        # Project to ambient space
        points = mean.unsqueeze(0) + torch.matmul(coeffs, basis.t())
        
        # Add noise
        noise = torch.randn(n_points_per_cluster, ambient_dim) * noise_level
        points = points + noise
        
        data_list.append(points)
        true_labels.extend([k] * n_points_per_cluster)
        
    # Combine all data
    X = torch.cat(data_list, dim=0)
    true_labels = torch.tensor(true_labels)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    true_labels = true_labels[perm]
    
    return X, true_labels


def plot_results(X, labels_kfactors, labels_kmeans, true_labels, kfactors_model):
    """Visualize clustering results using PCA projection."""
    
    # Use PCA to project to 3D for visualization
    X_centered = X - X.mean(dim=0)
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    X_pca = torch.matmul(X_centered, Vt.t()[:, :3])
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot true clusters
    ax1 = fig.add_subplot(131, projection='3d')
    for k in range(true_labels.max() + 1):
        mask = true_labels == k
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   alpha=0.6, s=30, label=f'Cluster {k}')
    ax1.set_title('True Clusters')
    ax1.legend()
    
    # Plot K-Factors results
    ax2 = fig.add_subplot(132, projection='3d')
    for k in range(labels_kfactors.max() + 1):
        mask = labels_kfactors == k
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   alpha=0.6, s=30, label=f'Cluster {k}')
    ax2.set_title('K-Factors Clustering')
    ax2.legend()
    
    # Plot K-means results
    ax3 = fig.add_subplot(133, projection='3d')
    for k in range(labels_kmeans.max() + 1):
        mask = labels_kmeans == k
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   alpha=0.6, s=30, label=f'Cluster {k}')
    ax3.set_title('K-Means Clustering')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot explained variance by each method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-Factors: variance explained per cluster
    variances_kfactors = []
    for k in range(kfactors_model.n_clusters):
        mask = labels_kfactors == k
        if mask.any():
            cluster_points = X[mask]
            # Compute reconstruction error
            centered = cluster_points - kfactors_model.cluster_centers_[k]
            basis = kfactors_model.cluster_bases_[k]
            coeffs = torch.matmul(centered, basis)
            reconstructed = torch.matmul(coeffs, basis.t())
            residual = centered - reconstructed
            
            total_var = torch.var(centered).item()
            residual_var = torch.var(residual).item()
            explained_var = 1 - (residual_var / total_var) if total_var > 0 else 0
            variances_kfactors.append(explained_var)
    
    ax1.bar(range(len(variances_kfactors)), variances_kfactors)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('K-Factors: Variance Explained per Cluster')
    ax1.set_ylim(0, 1)
    
    # Overall reconstruction quality
    X_transformed = kfactors_model.transform(X)
    X_reconstructed = kfactors_model.inverse_transform(X_transformed, labels_kfactors)
    
    reconstruction_error_kfactors = torch.mean((X - X_reconstructed) ** 2).item()
    
    # K-means doesn't have subspaces, so just compute distance to centroids
    kmeans_error = 0
    for k in range(labels_kmeans.max() + 1):
        mask = labels_kmeans == k
        if mask.any():
            cluster_points = X[mask]
            centroid = cluster_points.mean(dim=0)
            kmeans_error += torch.sum((cluster_points - centroid) ** 2).item()
    kmeans_error /= len(X)
    
    ax2.bar(['K-Factors', 'K-Means'], 
            [reconstruction_error_kfactors, kmeans_error])
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Reconstruction Error Comparison')
    
    plt.tight_layout()
    plt.show()


def compute_metrics(true_labels, pred_labels):
    """Compute clustering metrics."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    true_np = true_labels.numpy()
    pred_np = pred_labels.numpy()
    
    ari = adjusted_rand_score(true_np, pred_np)
    nmi = normalized_mutual_info_score(true_np, pred_np)
    
    return ari, nmi


def main():
    """Run the demo."""
    print("=== K-Factors Clustering Demo ===\n")
    
    # Generate synthetic data
    print("Generating synthetic data with local subspace structure...")
    X, true_labels = generate_subspace_data(
        n_points_per_cluster=200,
        n_clusters=3,
        ambient_dim=10,
        subspace_dim=2,
        noise_level=0.1
    )
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: {true_labels.max() + 1}\n")
    
    # Apply K-Factors
    print("Fitting K-Factors...")
    kfactors = KFactors(
        n_clusters=3,
        n_components=2,  # Match true subspace dimension
        penalty_weight=0.9,
        max_iter=20,
        verbose=1,
        random_state=42
    )
    kfactors.fit(X)
    labels_kfactors = kfactors.labels_
    
    print(f"\nK-Factors completed in {kfactors.n_iter_} iterations")
    print(f"Final objective: {kfactors.inertia_:.4f}")
    
    # Compare with K-means
    print("\nFitting K-Means for comparison...")
    kmeans = KMeans(
        n_clusters=3,
        max_iter=100,
        verbose=0,
        random_state=42
    )
    kmeans.fit(X)
    labels_kmeans = kmeans.labels_
    
    print(f"K-Means completed in {kmeans.n_iter_} iterations")
    print(f"Final objective: {kmeans.inertia_:.4f}")
    
    # Compute metrics
    print("\n=== Clustering Metrics ===")
    ari_kf, nmi_kf = compute_metrics(true_labels, labels_kfactors)
    ari_km, nmi_km = compute_metrics(true_labels, labels_kmeans)
    
    print(f"\nK-Factors:")
    print(f"  Adjusted Rand Index: {ari_kf:.3f}")
    print(f"  Normalized Mutual Info: {nmi_kf:.3f}")
    
    print(f"\nK-Means:")
    print(f"  Adjusted Rand Index: {ari_km:.3f}")
    print(f"  Normalized Mutual Info: {nmi_km:.3f}")
    
    # Show stage-wise progress
    print("\n=== K-Factors Stage Progress ===")
    for stage_info in kfactors.stage_history_:
        print(f"Stage {stage_info['stage'] + 1}: "
              f"{stage_info['iterations']} iterations, "
              f"objective = {stage_info['objective']:.4f}, "
              f"converged = {stage_info['converged']}")
    
    # Visualize results
    print("\nPlotting results...")
    plot_results(X, labels_kfactors, labels_kmeans, true_labels, kfactors)
    
    # Demonstrate transform/inverse_transform
    print("\n=== Transform/Reconstruction Demo ===")
    X_transformed = kfactors.transform(X[:5])  # Transform first 5 points
    X_reconstructed = kfactors.inverse_transform(X_transformed, labels_kfactors[:5])
    
    print(f"Original first point: {X[0, :5]}...")  # Show first 5 dims
    print(f"Transformed (local coords): {X_transformed[0]}")
    print(f"Reconstructed: {X_reconstructed[0, :5]}...")
    print(f"Reconstruction error: {torch.norm(X[0] - X_reconstructed[0]):.4f}")


if __name__ == "__main__":
    main()