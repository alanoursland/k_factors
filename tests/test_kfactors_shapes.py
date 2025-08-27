
import numpy as np
import pytest

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None

@pytest.mark.skipif(KFactors is None, reason="KFactors not implemented/exposed yet")
def test_kfactors_learns_subspace_bases_shapes():
    rng = np.random.default_rng(42)
    # Construct two clusters lying near different 1D lines in 3D
    t1 = rng.normal(size=(200, 1))
    t2 = rng.normal(size=(200, 1))
    c1 = np.hstack([t1, 0.1*rng.normal(size=(200,1)), 0.1*rng.normal(size=(200,1))])  # approx along x-axis
    c2 = np.hstack([0.1*rng.normal(size=(200,1)), t2, 0.1*rng.normal(size=(200,1))])  # approx along y-axis
    X = np.vstack([c1, c2])

    model = KFactors(n_clusters=2, n_components=1, random_state=0)
    model.fit(X)

    # Expect per-cluster 1D orthonormal bases
    assert hasattr(model, "cluster_bases_"), "Model should expose bases_"
    bases = model.cluster_bases_
    assert len(bases) == 2
    for B in bases:
        # B should be (n_features, n_components)
        assert B.shape[0] == X.shape[1]
        assert B.shape[1] == 1
        # Columns should be unit-norm
        col_norm = np.linalg.norm(B[:,0].cpu())
        print(col_norm)
        assert np.isclose(col_norm, 1.0, atol=1e-5)
