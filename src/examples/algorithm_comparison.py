"""
Comparison of different clustering algorithms in the K-Factors family.

This example demonstrates:
1. K-means (0D subspaces - just centroids)
2. K-Lines (1D subspaces)
3. K-Subspaces (2D subspaces)
4. K-Factors (sequential 2D subspaces with penalty)

Shows how reconstruction quality improves with richer representations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Add parent directory to path
import sys
sys.path.append('..')

from kfactors import KMeans, KLines, KSubspaces, KFactors, ClusteringBuilder


def generate_synthetic_data(n_samples=500, n_features=20, n_clusters=4, 
                          intrinsic_dim=3, noise_level=0.5, random_state=42):
    """Generate data with intrinsic low-dimensional structure.
    
    Each cluster has data concentrated around a different low-dimensional subspace.
    """
    torch.manual_seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    data_list = []
    true_labels = []
    
    for k in range(n_clusters):
        # Random subspace basis
        basis = torch.randn(n_features, intrinsic_dim)
        basis, _ = torch.linalg.qr(basis)
        
        # Random offset
        offset = torch.randn(n_features) * 5
        
        # Generate points in subspace
        latent = torch.randn(samples_per_cluster, intrinsic_dim) * 3
        points = offset + torch.matmul(latent, basis.t())
        
        # Add noise
        noise = torch.randn(samples_per_cluster, n_features) * noise_level
        points = points + noise
        
        data_list.append(points)
        true_labels.extend([k] * samples_per_cluster)
        
    # Combine and shuffle
    X = torch.cat(data_list, dim=0)
    true_labels = torch.tensor(true_labels)
    
    perm = torch.randperm(len(X))
    X = X[perm]
    true_labels = true_labels[perm]
    
    return X, true_labels


def evaluate_algorithm(algorithm, X, true_labels, name):
    """Fit algorithm and compute metrics."""
    print(f"\n{'='*50}")
    print(f"Testing {name}")
    print('='*50)
    
    # Time the fitting
    start_time = time()
    algorithm.fit(X)
    fit_time = time() - start_time
    
    # Get predictions
    labels = algorithm.predict(X)
    
    # Compute metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels.numpy(), labels.numpy())
    nmi = normalized_mutual_info_score(true_labels.numpy(), labels.numpy())
    
    # Reconstruction error (if applicable)
    recon_error = float('inf')
    if hasattr(algorithm, 'transform'):
        try:
            X_trans = algorithm.transform(X)
            X_recon = algorithm.inverse_transform(X_trans, labels)
            recon_error = torch.mean((X - X_recon) ** 2).item()
        except:
            pass
    elif hasattr(algorithm, 'project_to_lines'):
        try:
            X_proj = algorithm.project_to_lines(X)
            recon_error = torch.mean((X - X_proj) ** 2).item()
        except:
            pass
            
    # Inertia (objective value)
    inertia = algorithm.inertia_
    
    results = {
        'name': name,
        'time': fit_time,
        'iterations': algorithm.n_iter_,
        'ari': ari,
        'nmi': nmi,
        'inertia': inertia,
        'recon_error': recon_error
    }
    
    print(f"Fit time: {fit_time:.3f}s")
    print(f"Iterations: {algorithm.n_iter_}")
    print(f"ARI: {ari:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"Inertia: {inertia:.2f}")
    if recon_error < float('inf'):
        print(f"Reconstruction error: {recon_error:.4f}")
        
    return results


def plot_comparison(results_list):
    """Plot comparison of different algorithms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    names = [r['name'] for r in results_list]
    
    # 1. Clustering quality (ARI)
    ax = axes[0]
    aris = [r['ari'] for r in results_list]
    bars = ax.bar(names, aris)
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Clustering Quality')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, aris):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Information content (NMI)
    ax = axes[1]
    nmis = [r['nmi'] for r in results_list]
    bars = ax.bar(names, nmis)
    ax.set_ylabel('Normalized Mutual Information')
    ax.set_title('Information Preservation')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, nmis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Reconstruction error
    ax = axes[2]
    recon_errors = [r['recon_error'] for r in results_list if r['recon_error'] < float('inf')]
    recon_names = [r['name'] for r in results_list if r['recon_error'] < float('inf')]
    if recon_errors:
        bars = ax.bar(recon_names, recon_errors)
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Reconstruction Error')
        for bar, val in zip(bars, recon_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Computational time
    ax = axes[3]
    times = [r['time'] for r in results_list]
    bars = ax.bar(names, times)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Fitting Time')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}s', ha='center', va='bottom')
    
    # 5. Iterations
    ax = axes[4]
    iters = [r['iterations'] for r in results_list]
    bars = ax.bar(names, iters)
    ax.set_ylabel('Iterations')
    ax.set_title('Convergence Speed')
    for bar, val in zip(bars, iters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom')
    
    # 6. Objective value
    ax = axes[5]
    inertias = [r['inertia'] for r in results_list]
    bars = ax.bar(names, inertias)
    ax.set_ylabel('Objective Value')
    ax.set_title('Final Objective')
    
    # Rotate x-labels for all plots
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.show()


def main():
    """Run the comparison."""
    print("Generating synthetic data...")
    X, true_labels = generate_synthetic_data(
        n_samples=600,
        n_features=20,
        n_clusters=4,
        intrinsic_dim=3,
        noise_level=0.5
    )
    
    print(f"Data shape: {X.shape}")
    print(f"True clusters: {torch.unique(true_labels)}")
    
    # Common parameters
    n_clusters = 4
    max_iter = 50
    random_state = 42
    verbose = 0
    
    # Initialize algorithms
    algorithms = [
        (KMeans(n_clusters=n_clusters, max_iter=max_iter, 
                random_state=random_state, verbose=verbose), 
         "K-Means"),
         
        (KLines(n_clusters=n_clusters, max_iter=max_iter,
                random_state=random_state, verbose=verbose),
         "K-Lines (1D)"),
         
        (KSubspaces(n_clusters=n_clusters, n_components=2, max_iter=max_iter,
                    random_state=random_state, verbose=verbose),
         "K-Subspaces (2D)"),
         
        (KSubspaces(n_clusters=n_clusters, n_components=3, max_iter=max_iter,
                    random_state=random_state, verbose=verbose),
         "K-Subspaces (3D)"),
         
        (KFactors(n_clusters=n_clusters, n_components=3, max_iter=max_iter,
                  random_state=random_state, verbose=verbose),
         "K-Factors (3D)")
    ]
    
    # Test each algorithm
    results = []
    for algo, name in algorithms:
        result = evaluate_algorithm(algo, X, true_labels, name)
        results.append(result)
        
    # Plot comparison
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    plot_comparison(results)
    
    # Additional analysis: Builder pattern example
    print("\n" + "="*50)
    print("Testing custom algorithm with builder pattern")
    print("="*50)
    
    custom_algo = (ClusteringBuilder()
        .with_subspace_representation(dim=2)
        .with_hard_assignment()
        .with_pca_update(n_components=2)
        .with_kmeans_plusplus_init()
        .with_combined_convergence([
            ChangeInObjective(rel_tol=1e-4),
            ChangeInAssignments(min_change_fraction=0.001)
        ])
        .with_max_iter(max_iter)
        .with_random_state(random_state)
        .build(n_clusters=n_clusters))
    
    result = evaluate_algorithm(custom_algo, X, true_labels, "Custom (Builder)")
    
    print("\nComparison complete!")


if __name__ == "__main__":
    from kfactors.utils.convergence import ChangeInObjective, ChangeInAssignments
    main()