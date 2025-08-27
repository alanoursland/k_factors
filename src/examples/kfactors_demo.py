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
from sklearn.metrics import confusion_matrix


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

    # Make sure all label tensors are CPU for boolean indexing
    if hasattr(labels_kfactors, "detach"):
        labels_kfactors = labels_kfactors.detach().cpu()
    if hasattr(labels_kmeans, "detach"):
        labels_kmeans = labels_kmeans.detach().cpu()
    if hasattr(true_labels, "detach"):
        true_labels = true_labels.detach().cpu()
    X = X.detach().cpu() if hasattr(X, "detach") else X

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
            centered = cluster_points - kfactors_model.cluster_centers_[k].detach().cpu()
            basis = kfactors_model.cluster_bases_[k].detach().cpu()
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
    
    # Overall reconstruction quality (compute on the model's device, then print a CPU scalar)
    device = kfactors_model.device
    Xt_dev = X.to(device) if hasattr(X, "to") else torch.as_tensor(X, device=device)

    # labels_kfactors was normalized to CPU above; make a device copy for inverse_transform
    labels_dev = (
        labels_kfactors.to(device)
        if hasattr(labels_kfactors, "to")
        else torch.as_tensor(labels_kfactors, device=device)
    )

    X_transformed = kfactors_model.transform(Xt_dev)
    X_reconstructed_dev = kfactors_model.inverse_transform(X_transformed, labels_dev)

    reconstruction_error_kfactors = torch.mean((Xt_dev - X_reconstructed_dev) ** 2).item()
    
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
    """Compute clustering metrics (device-safe)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Ensure CPU + numpy
    true_np = true_labels.detach().cpu().numpy() if hasattr(true_labels, "detach") else true_labels
    pred_np = pred_labels.detach().cpu().numpy() if hasattr(pred_labels, "detach") else pred_labels

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
    # Ensure X is on the model's device for predict/objective
    Xt = X.to(kfactors.device)

    # Predict on-device; keep a CPU copy for plotting/metrics later
    labels_kfactors_dev = kfactors.predict(Xt)
    labels_kfactors = labels_kfactors_dev.detach().cpu()

    print(f"\nK-Factors completed in {kfactors.n_iter_} iterations")

    # Compute objective on-device so indexing devices match
    kf_obj = kfactors.objective.compute(Xt, kfactors.representations, labels_kfactors_dev).item()
    print(f"Final objective (computed): {kf_obj:.4f}")
    

    ###################################
    # print_diagnostics(kfactors, X, labels_kfactors, true_labels)
    print_features(kfactors, X, labels_kfactors)
    ###################################

    
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
    # Use the same Xt we already made (model.device) and on-device labels
    X_transformed = kfactors.transform(Xt[:5])
    X_reconstructed = kfactors.inverse_transform(X_transformed, labels_kfactors_dev[:5])
    
    print(f"Original first point: {X[0, :5]}...")  # Show first 5 dims
    print(f"Transformed (local coords): {X_transformed[0]}")
    print(f"Reconstructed: {X_reconstructed[0, :5]}...")
    err = torch.norm(X[0].to(X_reconstructed.device) - X_reconstructed[0]).item()
    print(f"Reconstruction error: {err:.4f}")

def print_diagnostics(kfactors, X, labels_kfactors, true_labels):
    print("\n=== Diagnostics: K-Factors assignments ===")

    # 0) Bring everything we need onto CPU for analysis/printing
    Xt_cpu = X.detach().cpu()
    labs_kf = labels_kfactors.detach().cpu().numpy()
    labs_true = true_labels.detach().cpu().numpy()

    # 1) Confusion matrix (perm-invariant alignment will happen via ARI/NMI,
    #    but this shows *which* two clusters are getting swapped/mixed)
    cm = confusion_matrix(labs_true, labs_kf)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # 2) Per-cluster counts
    unique, counts = np.unique(labs_kf, return_counts=True)
    print("Predicted counts per cluster:", dict(zip(unique.tolist(), counts.tolist())))

    # 3) Residual distance matrix D: for each point and cluster,
    #    squared orthogonal residual to that cluster’s *final* subspace (all R dims).
    with torch.no_grad():
        D_cols = []
        for k, rep in enumerate(kfactors.representations):
            centered = Xt_cpu - rep.mean.detach().cpu().unsqueeze(0)
            if hasattr(rep, "W"):
                B = rep.W.detach().cpu()         # (d, R)
            elif hasattr(rep, "basis"):
                B = rep.basis.detach().cpu()     # (d, R)
            else:
                # fallback: no subspace, use pure centroid distance
                B = torch.empty(rep.dimension, 0)

            if B.shape[1] > 0:
                coeffs = centered @ B            # (n, R)
                proj = coeffs @ B.T              # (n, d)
                resid = centered - proj
            else:
                resid = centered
            D_cols.append((resid * resid).sum(dim=1, keepdim=True))  # (n, 1)

        D = torch.cat(D_cols, dim=1)  # (n, K)

    # 4) Margin analysis: how close is each point to the second-best cluster?
    assigned = torch.as_tensor(labs_kf)
    top = D[torch.arange(len(D)), assigned]
    D_masked = D.clone()
    D_masked[torch.arange(len(D)), assigned] = float("inf")
    second = D_masked.min(dim=1).values
    margin = (second - top) / (second + 1e-12)  # ∈ [0,1], bigger is safer

    q = torch.quantile(margin, torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75]))
    print("Margin quantiles (1%,5%,10%,25%,50%,75%):", [float(x) for x in q])
    print("Fraction with tiny margin (< 0.02):", float((margin < 0.02).float().mean()))

    # 5) Look specifically at the two troubled clusters (guess they’re 0 and 2 in your plot).
    #    Change these indices to whichever pair looks mixed in your PCA figure.
    a, b = 0, 2
    mask_ab = (labs_true == a) | (labs_true == b)
    cm_ab = confusion_matrix(labs_true[mask_ab], labs_kf[mask_ab], labels=[a, b])
    print(f"Confusion submatrix for true clusters {a} & {b}:\n", cm_ab)

    # 6) How “parallel” are the final subspaces for those two clusters?
    #    Use principal angles between the 2D bases (R=2). If they’re nearly parallel,
    #    assignments can be ambiguous in a projection.
    def principal_angles(B1, B2, eps=1e-12):
        # expects (d, r) with orthonormal columns
        U, _, Vt = torch.linalg.svd(B1.T @ B2)
        s = torch.clamp(torch.diag(_ if _.dim()==2 else torch.diag_embed(_)), 0, 1) if False else torch.diag(torch.eye(min(B1.shape[1], B2.shape[1])))
        # simpler: singular values of B1^T B2 are cosines of angles
        cosines = torch.linalg.svd(B1.T @ B2, full_matrices=False).S.clamp(-1,1)
        return torch.arccos(cosines)  # radians

    if hasattr(kfactors.representations[a], "W"):
        Ba = kfactors.representations[a].W.detach().cpu()
    else:
        Ba = kfactors.representations[a].basis.detach().cpu()

    if hasattr(kfactors.representations[b], "W"):
        Bb = kfactors.representations[b].W.detach().cpu()
    else:
        Bb = kfactors.representations[b].basis.detach().cpu()

    ang = principal_angles(Ba, Bb)
    print(f"Principal angles between cluster {a} and {b} bases (radians):",
        [float(x) for x in ang])
    print(f"(Degrees):", [float(x*180.0/np.pi) for x in ang])

def print_features(kfactors, X, labels_kfactors):
    # === K-Factors learned features (means, bases, per-component capture) ===
    print("\n=== K-Factors learned features ===")

    with torch.no_grad():
        # Always analyze on CPU for clean printing
        Xc = X.detach().cpu()
        labs_kf = labels_kfactors.detach().cpu()
        K = kfactors.n_clusters

        def get_basis(rep):
            if hasattr(rep, "W"):
                return rep.W.detach().cpu()            # (d, R)
            elif hasattr(rep, "basis"):
                return rep.basis.detach().cpu()        # (d, R)
            else:
                return torch.empty(rep.dimension, 0)

        # 1) Per-cluster summary
        for k in range(K):
            rep = kfactors.representations[k]
            mu = rep.mean.detach().cpu()
            B = get_basis(rep)                         # (d, R)
            d, R = (B.shape[0], B.shape[1]) if B.numel() > 0 else (mu.numel(), 0)

            print(f"\n[Cluster {k}] mean (first 5 dims): {mu[:5].numpy()}")
            if R == 0:
                print("  (No basis)")
                continue

            # Norms and orthogonality
            col_norms = torch.norm(B, dim=0)
            gram = (B.t() @ B)                         # should be ~ I if orthonormal
            print(f"  basis shape: {tuple(B.shape)}")
            print(f"  column norms: {col_norms.numpy().round(6)}")
            print(f"  V^T V (rounded):\n{gram.numpy().round(4)}")

            # Top-5 absolute loadings per component
            for r in range(R):
                vr = B[:, r]
                top5_idx = torch.topk(vr.abs(), k=min(5, d)).indices.tolist()
                top5 = [(i, float(vr[i])) for i in top5_idx]
                print(f"  v[{r}] top-5 |loadings| (index, value): {top5}")

            # Per-component variance capture on *assigned* points
            mask = (labs_kf == k)
            if mask.any():
                Xk = Xc[mask]
                centered = Xk - mu.unsqueeze(0)
                coeffs = centered @ B                  # (n_k, R)
                total_energy = float((centered**2).sum().item()) + 1e-12
                cap = (coeffs**2).sum(dim=0)           # energy per component
                frac = (cap / total_energy).numpy()
                print(f"  per-component capture on assigned points: {np.round(frac, 4)}")
                print(f"  cumulative capture: {float(frac.sum()):.4f}")
            else:
                print("  (No points assigned)")

        # 2) Cross-cluster cosine tables per stage (who looks like whom?)
        #    For each r, compute |cos| between v_k[:,r] across clusters.
        repsB = [get_basis(rep) for rep in kfactors.representations]
        if all(B.numel() > 0 for B in repsB):
            R = repsB[0].shape[1]
            for r in range(R):
                M = torch.zeros(K, K)
                for i in range(K):
                    vi = repsB[i][:, r]
                    vi = vi / (vi.norm() + 1e-12)
                    for j in range(K):
                        vj = repsB[j][:, r]
                        vj = vj / (vj.norm() + 1e-12)
                        M[i, j] = torch.abs(torch.dot(vi, vj))
                print(f"\n|cos| between clusters for component r={r}:\n{M.numpy().round(4)}")


if __name__ == "__main__":
    main()