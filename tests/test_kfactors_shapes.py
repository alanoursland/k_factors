
import numpy as np
import pytest

try:
    from kfactors.algorithms import KFactors
except Exception:
    KFactors = None

@pytest.mark.skipif(KFactors is None, reason="KFactors not implemented/exposed yet")
def test_kfactors_finds_local_directions_and_separates_clusters():
    rng = np.random.default_rng(42)
    # two 1D-ish clusters in 3D: along x-axis and y-axis (plus small noise)
    t1 = rng.normal(size=(200, 1))
    t2 = rng.normal(size=(200, 1))
    c1 = np.hstack([t1, 0.1*rng.normal(size=(200,1)), 0.1*rng.normal(size=(200,1))])  # ~ex
    c2 = np.hstack([0.1*rng.normal(size=(200,1)), t2, 0.1*rng.normal(size=(200,1))])  # ~ey
    X = np.vstack([c1, c2]).astype(np.float32)

    model = KFactors(n_clusters=2, n_components=1, random_state=0)
    model.fit(X)

    # ---- check learned directions align with {ex, ey} up to sign/perm/scale ----
    assert hasattr(model, "cluster_bases_")
    bases = model.cluster_bases_  # (K, d, r) torch tensor; here r=1
    assert bases.shape[0] == 2 and bases.shape[1] == X.shape[1] and bases.shape[2] == 1

    # normalize per-cluster vectors (scale not guaranteed in PPCA W)
    B = []
    for k in range(2):
        v = bases[k, :, 0].detach().cpu().numpy()
        v = v / (np.linalg.norm(v) + 1e-12)
        B.append(v)
    B = np.stack(B, axis=0)  # (2, d)

    G = np.eye(3)[:2]  # ground-truth directions ex, ey → shape (2, 3)

    # pairwise |cosine| similarities (2x2)
    S = np.abs(B @ G.T)
    # best permutation-invariant match score
    match_score = max(S[0,0] + S[1,1], S[0,1] + S[1,0])
    # require strong alignment to the two axes (both ~>0.9 on average)
    assert match_score > 1.8

    # ---- check the clusters are actually separated (label permutation-safe) ----
    y_pred = model.labels_.detach().cpu().numpy()
    first_half = y_pred[:200]
    second_half = y_pred[200:]

    # try both mappings (labels may be swapped)
    acc_mapA = (np.sum(first_half == 0) + np.sum(second_half == 1)) / y_pred.size
    acc_mapB = (np.sum(first_half == 1) + np.sum(second_half == 0)) / y_pred.size
    acc = max(acc_mapA, acc_mapB)

    assert acc > 0.9
