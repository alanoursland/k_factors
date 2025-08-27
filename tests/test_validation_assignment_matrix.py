# tests/unit/test_validation_assignment_matrix.py
"""
U6 — Validation & AssignmentMatrix conversions (Phase 1)

Covers:
- Base algorithm `_validate_data` converts list / numpy → torch.float32 on the right device
- AssignmentMatrix hard↔soft conversions and counts
- Threshold behavior of `get_cluster_indices` for soft assignments
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from kfactors.base.data_structures import AssignmentMatrix

# Access a concrete algorithm to call _validate_data (protected); KFactors is fine
try:
    from kfactors.algorithms import KFactors  # type: ignore
except Exception:  # pragma: no cover
    KFactors = None  # type: ignore


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


@pytest.mark.skipif(KFactors is None, reason="KFactors not importable")
def test_validate_data_numpy_and_list_to_tensor(seed_all, torch_device):
    device = torch_device
    model = KFactors(n_clusters=2, n_components=1, random_state=0, device=device)

    # numpy array (float64) → torch.float32 on model.device
    X_np = np.random.RandomState(0).randn(5, 3).astype(np.float64)
    X_t = model._validate_data(X_np)  # protected method by design in tests
    assert isinstance(X_t, torch.Tensor)
    assert X_t.dtype == torch.float32
    assert X_t.device.type == device.type
    assert X_t.shape == (5, 3)

    # python list of lists → torch.float32 tensor
    X_list = [[1.0, 2.0, 3.0], [0.5, -0.1, 4.2]]
    X_t2 = model._validate_data(X_list)
    assert isinstance(X_t2, torch.Tensor)
    assert X_t2.dtype == torch.float32
    assert X_t2.device.type == device.type
    assert X_t2.shape == (2, 3)


def test_assignment_matrix_hard_to_soft_and_back(seed_all, torch_device):
    device = torch_device
    hard = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
    K = 2

    # Hard → Soft
    AM_hard = AssignmentMatrix(hard, n_clusters=K, is_soft=False)
    soft = AM_hard.get_soft()
    assert soft.shape == (4, 2)
    # one-hot rows
    expected_soft = torch.tensor(
        [[1, 0], [0, 1], [1, 0], [0, 1]], device=device, dtype=soft.dtype
    )
    assert torch.allclose(soft, expected_soft)

    # Counts
    counts_hard = AM_hard.count_per_cluster()
    assert torch.allclose(counts_hard, torch.tensor([2.0, 2.0], device=device))

    # Soft → Hard (argmax)
    AM_soft = AssignmentMatrix(soft, n_clusters=K, is_soft=True)
    hard_back = AM_soft.get_hard()
    assert torch.equal(hard_back, hard)

    counts_soft = AM_soft.count_per_cluster()
    assert torch.allclose(counts_soft, torch.tensor([2.0, 2.0], device=device))


def test_assignment_matrix_get_cluster_indices_threshold_behavior(seed_all, torch_device):
    device = torch_device
    # Responsibilities; threshold is strictly > 0.5 in implementation
    soft = torch.tensor(
        [
            [0.60, 0.40],  # → 0
            [0.40, 0.60],  # → 1
            [0.51, 0.49],  # → 0
            [0.49, 0.51],  # → 1
        ],
        device=device,
        dtype=torch.float32,
    )
    AM = AssignmentMatrix(soft, n_clusters=2, is_soft=True)

    idx0 = AM.get_cluster_indices(0)
    idx1 = AM.get_cluster_indices(1)

    # Convert to Python sets for order-agnostic comparison
    assert set(idx0.tolist()) == {0, 2}
    assert set(idx1.tolist()) == {1, 3}
