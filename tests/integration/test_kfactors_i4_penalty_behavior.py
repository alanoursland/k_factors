import numpy as np
import pytest
import torch

from utils import time_block
from kfactors.assignments.penalized import PenalizedAssignment
from kfactors.base.data_structures import DirectionTracker
from kfactors.representations.ppca import PPCARepresentation


def _ppca_1d_with_dir(d: int, v: np.ndarray, device: torch.device) -> PPCARepresentation:
    """
    Make a PPCA rep with latent_dim=1 and basis aligned to v (unit-normalized).
    Mean is zero.
    """
    v = np.asarray(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    rep = PPCARepresentation(dimension=d, latent_dim=1, device=device, init_variance=1.0)
    with torch.no_grad():
        rep.mean = torch.zeros(d, device=device, dtype=torch.float32)
        W = torch.zeros(d, 1, device=device, dtype=torch.float32)
        W[:, 0] = torch.tensor(v, device=device, dtype=torch.float32)
        rep.W = W
    return rep


@pytest.mark.parametrize("penalty_weight", [1.0])  # penalty_weight multiplies the penalty; keep at full strength for clarity
def test_i4_penalty_aligned_vs_orthogonal_end_to_end_product(seed_all, torch_device, penalty_weight):
    """
    Product mode, single prior claim with weight=1.0:
      - candidate aligned with claimed dir => penalty ≈ 0.0
      - candidate orthogonal to claimed   => penalty ≈ 1.0
    """
    d = 2
    K = 2
    n = 1
    device = torch_device

    e0 = np.array([1.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0], dtype=np.float32)

    # One point at the origin (distances tie; assignments will pick index 0 by argmin)
    X = torch.zeros(n, d, device=device, dtype=torch.float32)

    reps = [
        _ppca_1d_with_dir(d, e0, device),  # cluster 0: aligned with prior
        _ppca_1d_with_dir(d, e1, device),  # cluster 1: orthogonal to prior
    ]

    tracker = DirectionTracker(n_points=n, n_clusters=K, device=device)
    # Prior claim: along e0 with full weight
    tracker.add_claimed_direction(point_idx=0, cluster_idx=0, direction=torch.tensor(e0, device=device), weight=1.0)

    assigner = PenalizedAssignment(penalty_type="product", penalty_weight=penalty_weight)

    with time_block("I4-product", meta={"n": n, "K": K, "R": 1}):
        assignments, aux = assigner.compute_assignments(
            X, reps, direction_tracker=tracker, current_stage=0
        )

    penalties = aux["penalties"].detach().cpu().numpy()  # (1, 2)
    p_aligned, p_orth = float(penalties[0, 0]), float(penalties[0, 1])

    assert np.isclose(p_aligned, 0.0, atol=0.02), f"aligned penalty should ~0.0; got {p_aligned:.3f}"
    assert np.isclose(p_orth, 1.0, atol=0.02), f"orth penalty should ~1.0; got {p_orth:.3f}"


@pytest.mark.parametrize("penalty_weight", [1.0])
def test_i4_penalty_numeric_match_weighted_product(seed_all, torch_device, penalty_weight):
    """
    Product mode with partial prior weight=0.7:
      aligned  : 1 - 0.7*1 = 0.3
      orthogonal: 1 - 0.7*0 = 1.0
    """
    d = 2
    K = 2
    n = 1
    device = torch_device

    e0 = np.array([1.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0], dtype=np.float32)

    X = torch.zeros(n, d, device=device, dtype=torch.float32)

    reps = [
        _ppca_1d_with_dir(d, e0, device),  # aligned
        _ppca_1d_with_dir(d, e1, device),  # orthogonal
    ]

    tracker = DirectionTracker(n_points=n, n_clusters=K, device=device)
    tracker.add_claimed_direction(0, 0, torch.tensor(e0, device=device), weight=0.7)

    assigner = PenalizedAssignment(penalty_type="product", penalty_weight=penalty_weight)

    with time_block("I4-weighted-product", meta={"n": n, "K": K, "R": 1}):
        assignments, aux = assigner.compute_assignments(
            X, reps, direction_tracker=tracker, current_stage=0
        )

    penalties = aux["penalties"].detach().cpu().numpy()
    p_aligned, p_orth = float(penalties[0, 0]), float(penalties[0, 1])

    assert np.isclose(p_aligned, 0.3, atol=0.02), f"aligned penalty should ~0.3; got {p_aligned:.3f}"
    assert np.isclose(p_orth, 1.0, atol=0.02), f"orth penalty should ~1.0; got {p_orth:.3f}"


@pytest.mark.parametrize("penalty_weight", [1.0])
def test_i4_penalty_mean_mode_numeric(seed_all, torch_device, penalty_weight):
    """
    Mean ('sum') mode with partial prior weight=0.7:
      aligned   : 1 - (0.7*1 / 0.7) = 0.0
      orthogonal: 1 - (0.7*0 / 0.7) = 1.0
    """
    d = 2
    K = 2
    n = 1
    device = torch_device

    e0 = np.array([1.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0], dtype=np.float32)

    X = torch.zeros(n, d, device=device, dtype=torch.float32)

    reps = [
        _ppca_1d_with_dir(d, e0, device),  # aligned
        _ppca_1d_with_dir(d, e1, device),  # orthogonal
    ]

    tracker = DirectionTracker(n_points=n, n_clusters=K, device=device)
    tracker.add_claimed_direction(0, 0, torch.tensor(e0, device=device), weight=0.7)

    assigner = PenalizedAssignment(penalty_type="sum", penalty_weight=penalty_weight)

    with time_block("I4-mean-mode", meta={"n": n, "K": K, "R": 1}):
        assignments, aux = assigner.compute_assignments(
            X, reps, direction_tracker=tracker, current_stage=0
        )

    penalties = aux["penalties"].detach().cpu().numpy()
    p_aligned, p_orth = float(penalties[0, 0]), float(penalties[0, 1])

    assert np.isclose(p_aligned, 0.0, atol=0.02), f"aligned penalty should ~0.0; got {p_aligned:.3f}"
    assert np.isclose(p_orth, 1.0, atol=0.02), f"orth penalty should ~1.0; got {p_orth:.3f}"


@pytest.mark.parametrize("penalty_weight", [1.0])
def test_i4_effective_sample_sizes_log_product(seed_all, torch_device, penalty_weight):
    """
    Show how per-cluster effective sample sizes reflect the penalty via gather on assignments.
    Design points so assignments split deterministically by geometry.

    Setup:
      - points 0,1 → small along e0  (assign to cluster 0), prior (e0, 1.0) → aligned penalty ~0.0 each
      - points 2,3 → small along e0  (assign to cluster 0), prior (e0, 0.7) → aligned penalty ~0.3 each
      - points 4,5 → small along e1  (assign to cluster 1), prior (e0, 0.7) → orth penalty ~1.0 each

    Expected:
      eff_n_0 ≈ 0.0 + 0.0 + 0.3 + 0.3 = 0.6
      eff_n_1 ≈ 1.0 + 1.0 = 2.0
    """
    d = 2
    K = 2
    device = torch_device

    e0 = np.array([1.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0], dtype=np.float32)

    eps = 1e-3
    X_np = np.stack(
        [
            eps * e0,  # p0 → cl0
            (2 * eps) * e0,  # p1 → cl0
            (3 * eps) * e0,  # p2 → cl0
            (4 * eps) * e0,  # p3 → cl0
            eps * e1,  # p4 → cl1
            (2 * eps) * e1,  # p5 → cl1
        ],
        axis=0,
    ).astype(np.float32)
    X = torch.tensor(X_np, device=device)

    reps = [
        _ppca_1d_with_dir(d, e0, device),  # cluster 0 basis = e0
        _ppca_1d_with_dir(d, e1, device),  # cluster 1 basis = e1
    ]

    tracker = DirectionTracker(n_points=X.shape[0], n_clusters=K, device=device)
    # p0, p1: full prior along e0
    tracker.add_claimed_direction(0, 0, torch.tensor(e0, device=device), weight=1.0)
    tracker.add_claimed_direction(1, 0, torch.tensor(e0, device=device), weight=1.0)
    # p2, p3: partial prior along e0 (0.7) — still aligned with cluster 0
    tracker.add_claimed_direction(2, 0, torch.tensor(e0, device=device), weight=0.7)
    tracker.add_claimed_direction(3, 0, torch.tensor(e0, device=device), weight=0.7)
    # p4, p5: partial prior along e0 (0.7) — but they will assign to cluster 1 (orthogonal candidate)
    tracker.add_claimed_direction(4, 0, torch.tensor(e0, device=device), weight=0.7)
    tracker.add_claimed_direction(5, 0, torch.tensor(e0, device=device), weight=0.7)

    assigner = PenalizedAssignment(penalty_type="product", penalty_weight=penalty_weight)

    with time_block("I4-effective-n", meta={"n": int(X.shape[0]), "K": K, "R": 1}):
        assignments, aux = assigner.compute_assignments(
            X, reps, direction_tracker=tracker, current_stage=0
        )

    penalties = aux["penalties"]  # (n, K), torch
    # per-point stage weight = penalty of its assigned cluster
    stage_weights = penalties.gather(1, assignments.view(-1, 1)).squeeze(1)

    # Sum by cluster
    eff_n_0 = float(stage_weights[assignments == 0].sum().item())
    eff_n_1 = float(stage_weights[assignments == 1].sum().item())

    # Expected values with small tolerance
    assert np.isclose(eff_n_0, 0.6, atol=0.05), f"eff_n_0 mismatch: expected ~0.6, got {eff_n_0:.3f}"
    assert np.isclose(eff_n_1, 2.0, atol=0.05), f"eff_n_1 mismatch: expected ~2.0, got {eff_n_1:.3f}"
