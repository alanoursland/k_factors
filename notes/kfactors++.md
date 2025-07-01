```
class KFactorsPlusPlus:
    def __init__(self, n_runs=10, dbscan_eps=0.1, min_samples=3):
        self.n_runs = n_runs
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples
    
    def fit(self, X):
        # Collect hyperplanes from multiple runs
        all_hyperplanes = []
        for run in range(self.n_runs):
            kf = KFactors(n_clusters=k, n_components=r)
            kf.fit(X)
            
            # Extract hyperplanes (W vectors normalized)
            for cluster_bases in kf.cluster_bases_:
                for basis_vector in cluster_bases:
                    all_hyperplanes.append(basis_vector / basis_vector.norm())
        
        # Cluster hyperplanes in parameter space
        from sklearn.cluster import DBSCAN
        hyperplane_clusters = DBSCAN(eps=self.dbscan_eps, 
                                    min_samples=self.min_samples)
        hyperplane_clusters.fit(all_hyperplanes)
        
        # Extract consensus surfaces (cluster centers)
        consensus_surfaces = []
        for label in np.unique(hyperplane_clusters.labels_):
            if label != -1:  # Ignore noise
                cluster_mask = hyperplane_clusters.labels_ == label
                consensus = np.mean(all_hyperplanes[cluster_mask], axis=0)
                consensus /= np.linalg.norm(consensus)
                consensus_surfaces.append(consensus)
        
        return consensus_surfaces
```