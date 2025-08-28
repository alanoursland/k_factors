
import numpy as np
import pytest
import torch

try:
    from kfactors.algorithms import KFactors
except Exception:  # pragma: no cover
    KFactors = None


def _abs_cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    u = u / (u.norm() + eps)
    v = v / (v.norm() + eps)
    return float(torch.abs(torch.dot(u, v)).item())


@pytest.mark.skipif(KFactors is None, reason="KFactors not implemented/exposed yet")
def test_kfactors_finds_local_directions_and_separates_clusters():
    rng = np.random.default_rng(42)
    # two 1D-ish clusters in 3D: along x-axis and y-axis (plus small noise)
    t1 = rng.normal(size=(200, 1))
    t2 = rng.normal(size=(200, 1))
    c1 = np.hstack([t1, 0.1*rng.normal(size=(200,1)), 0.1*rng.normal(size=(200,1))])  # ~ex
    c2 = np.hstack([0.1*rng.normal(size=(200,1)), t2, 0.1*rng.normal(size=(200,1))])  # ~ey
    X = np.vstack([c1, c2]).astype(np.float32)

    model = KFactors(n_clusters=2, n_components=1, random_state=0, verbose=0)
    model.fit(X)

    # With the flat EigenFactorRepresentation, learned directions are in cluster_vectors_: (K, D)
    W = model.cluster_vectors_.detach().cpu()  # shape (2, 3)

    ex = torch.tensor([1.0, 0.0, 0.0])
    ey = torch.tensor([0.0, 1.0, 0.0])

    # similarity matrix between learned vectors and canonical axes
    S = torch.empty(2, 2)
    S[0, 0] = _abs_cos(W[0], ex)
    S[0, 1] = _abs_cos(W[0], ey)
    S[1, 0] = _abs_cos(W[1], ex)
    S[1, 1] = _abs_cos(W[1], ey)

    # best bipartite match (2x2 case): either (0->ex, 1->ey) or (0->ey, 1->ex)
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
