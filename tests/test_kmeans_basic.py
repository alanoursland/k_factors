
import numpy as np
import pytest

try:
    from kfactors.algorithms import KMeans
except Exception:
    KMeans = None

@pytest.mark.skipif(KMeans is None, reason="KMeans not implemented/exposed yet")
def test_kmeans_fits_simple_blobs():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=0.0, scale=0.3, size=(100, 2))
    X2 = rng.normal(loc=3.0, scale=0.3, size=(100, 2))
    X = np.vstack([X1, X2])

    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)

    assert hasattr(km, "labels_"), "Expected labels_ after fit"
    assert len(km.labels_) == X.shape[0]
    assert getattr(km, "cluster_centers_", None) is not None
