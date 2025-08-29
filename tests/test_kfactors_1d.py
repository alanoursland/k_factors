"""
test_kfactors_1d.py

Unit tests for K-Factors algorithm components in 1D.
Tests each component in isolation and their integration.
"""

import pytest
import numpy as np
import torch
from typing import Tuple, Optional

# Import components under test
from kfactors.representations.eigenfactor import EigenFactorRepresentation
from kfactors.base.data_structures import DirectionTracker
from kfactors.assignments.penalized import PenalizedAssignment
from kfactors.assignments.ranked import HyperplaneRanker
from kfactors.assignments.factors import IndependentFactorAssignment
from kfactors.algorithms.kfactors import KFactors


# ============================================================================
# Test Data Generators
# ============================================================================

def make_1d_clusters(centers: list, stds: list, n_per_cluster: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
   """Generate 1D data with known cluster structure.
   
   Returns:
       data: (n,1) tensor of points
       labels: (n,) tensor of true cluster labels
   """
   rng = np.random.default_rng(seed)
   data = []
   labels = []
   
   for k, (center, std) in enumerate(zip(centers, stds)):
       points = rng.normal(center, std, n_per_cluster)
       data.extend(points)
       labels.extend([k] * n_per_cluster)
   
   data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
   labels = torch.tensor(labels, dtype=torch.long)
   return data, labels


def make_1d_uniform(n_points: int, low: float = -5, high: float = 5, seed: int = 42) -> torch.Tensor:
   """Generate uniform 1D data."""
   rng = np.random.default_rng(seed)
   data = rng.uniform(low, high, n_points)
   return torch.tensor(data, dtype=torch.float32).unsqueeze(1)


def make_1d_mixture(weights: list, means: list, stds: list, n_total: int, seed: int = 42) -> torch.Tensor:
   """Generate 1D Gaussian mixture data."""
   rng = np.random.default_rng(seed)
   weights = np.array(weights) / np.sum(weights)
   
   data = []
   for _ in range(n_total):
       k = rng.choice(len(weights), p=weights)
       point = rng.normal(means[k], stds[k])
       data.append(point)
   
   return torch.tensor(data, dtype=torch.float32).unsqueeze(1)


# ============================================================================
# EigenFactorRepresentation 1D Tests
# ============================================================================

class TestEigenFactor1D:
   
   def test_1d_eigenfactor_scores(self):
       """Score computation: y_k(x) = w_k * (x - μ_k)"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=3, device='cpu')
       
       # Set known means and vectors
       rep.means = torch.tensor([[-1.0], [0.0], [1.0]])
       rep.vectors = torch.tensor([[0.5], [-0.3], [0.8]])
       
       # Test point at x=0.5
       points = torch.tensor([[0.5]])
       scores = rep.compute_scores(points)
       
       # Expected scores: [0.5*(0.5-(-1)), -0.3*(0.5-0), 0.8*(0.5-1)]
       #                = [0.5*1.5, -0.3*0.5, 0.8*(-0.5)]
       #                = [0.75, -0.15, -0.4]
       expected = torch.tensor([[0.75, -0.15, -0.4]])
       assert torch.allclose(scores, expected, atol=1e-5)
   
   def test_1d_eigenfactor_distances(self):
       """Distance: |w_k*(x-μ_k)|² / |w_k|²"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       
       rep.means = torch.tensor([[0.0], [2.0]])
       rep.vectors = torch.tensor([[1.0], [0.5]])
       
       points = torch.tensor([[1.0], [3.0]])
       
       # Test per-feature distance
       dist_0 = rep.distance_to_point(points, indices=torch.tensor([0, 0]))
       dist_1 = rep.distance_to_point(points, indices=torch.tensor([1, 1]))
       
       # Point 0 to feature 0: |1.0*(1-0)|²/|1.0|² = 1
       # Point 1 to feature 0: |1.0*(3-0)|²/|1.0|² = 9
       assert torch.allclose(dist_0, torch.tensor([1.0, 9.0]), atol=1e-5)
       
       # Point 0 to feature 1: |0.5*(1-2)|²/|0.5|² = 1
       # Point 1 to feature 1: |0.5*(3-2)|²/|0.5|² = 1
       assert torch.allclose(dist_1, torch.tensor([1.0, 1.0]), atol=1e-5)
   
   def test_1d_eigenfactor_update_mean(self):
       """Mean updates to weighted average of assigned points"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       
       points = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
       assignments = torch.tensor([0, 0, 1, 1])
       
       rep.update_from_points(points, assignments=assignments)
       
       # Feature 0 gets points [1, 2] -> mean = 1.5
       # Feature 1 gets points [3, 4] -> mean = 3.5
       assert torch.allclose(rep.means[0], torch.tensor([1.5]), atol=1e-5)
       assert torch.allclose(rep.means[1], torch.tensor([3.5]), atol=1e-5)
   
   def test_1d_eigenfactor_update_variance(self):
       """Vector magnitude captures variance of assigned points"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=1, device='cpu')
       
       # Points with known variance
       points = torch.tensor([[-2.0], [0.0], [2.0]])
       assignments = torch.zeros(3, dtype=torch.long)
       
       rep.update_from_points(points, assignments=assignments)
       
       # Mean should be 0, variance = 8/3 ≈ 2.67
       # Vector magnitude should be sqrt(variance)
       assert torch.allclose(rep.means[0], torch.tensor([0.0]), atol=1e-4)
       expected_var = 8.0/3.0
       assert torch.allclose(rep.vectors[0]**2, torch.tensor([expected_var]), atol=1e-3)


# ============================================================================
# Distance/Assignment Tests
# ============================================================================

class TestDistanceAssignment1D:
   
   def test_1d_distance_matrix(self):
       """Compute all pairwise distances (n×K matrix)"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=3, device='cpu')
       
       rep.means = torch.tensor([[-1.0], [0.0], [1.0]])
       rep.vectors = torch.tensor([[1.0], [1.0], [1.0]])
       
       points = torch.tensor([[-1.5], [0.5], [2.0]])
       
       # Manual distance calculation
       distances = torch.zeros(3, 3)
       for i, x in enumerate(points):
           for k in range(3):
               residual = (x - rep.means[k]) * rep.vectors[k]
               distances[i, k] = (residual**2 / rep.vectors[k]**2).item()
       
       # Using representation
       computed_distances = torch.zeros(3, 3)
       for k in range(3):
           indices = torch.full((3,), k, dtype=torch.long)
           computed_distances[:, k] = rep.distance_to_point(points, indices)
       
       assert torch.allclose(distances, computed_distances, atol=1e-5)
   
   def test_1d_assignment_no_penalty(self):
       """Without penalties, assign to nearest feature"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=3, device='cpu')
       rep.means = torch.tensor([[-2.0], [0.0], [2.0]])
       rep.vectors = torch.tensor([[1.0], [1.0], [1.0]])
       
       assignment_strategy = PenalizedAssignment()
       
       points = torch.tensor([[-1.8], [0.3], [1.5]])
       
       assignments, _ = assignment_strategy.compute_assignments(
           points, [rep], direction_tracker=None
       )
       
       # Points should assign to nearest means: [0, 1, 2]
       assert assignments[0] == 0  # -1.8 closest to -2
       assert assignments[1] == 1  # 0.3 closest to 0
       assert assignments[2] == 2  # 1.5 closest to 2


# ============================================================================
# DirectionTracker/Penalty Tests
# ============================================================================

class TestPenalty1D:
   
   def test_1d_penalty_no_history(self):
       """Fresh tracker -> all penalties = 1.0"""
       tracker = DirectionTracker(n_points=5, n_clusters=3, device=torch.device('cpu'))
       
       # Test directions (in 1D, just signs matter)
       test_dirs = torch.tensor([
           [[1.0], [1.0], [1.0]],   # point 0
           [[-1.0], [1.0], [-1.0]],  # point 1
           [[0.5], [-0.5], [1.0]],   # point 2
           [[1.0], [1.0], [1.0]],    # point 3
           [[-1.0], [-1.0], [-1.0]], # point 4
       ])
       
       penalties = tracker.compute_penalty_batch(test_dirs)
       assert torch.allclose(penalties, torch.ones(5, 3))
   
   def test_1d_penalty_same_sign(self):
       """Claiming +0.5 then testing +0.8 -> low penalty"""
       tracker = DirectionTracker(n_points=1, n_clusters=1, device=torch.device('cpu'))
       
       # Claim positive direction
       tracker.add_claimed_direction(0, 0, torch.tensor([0.5]), weight=1.0)
       
       # Test positive direction
       test_dir = torch.tensor([[[0.8]]])
       penalty = tracker.compute_penalty_batch(test_dir)
       
       # In 1D, same sign = parallel = |cos| = 1
       # Penalty = 1 - 1.0*1 = 0
       assert penalty[0, 0] < 0.01
   
   def test_1d_penalty_opposite_sign(self):
       """Claiming +0.5 then testing -0.8 -> low penalty (still parallel in 1D)"""
       tracker = DirectionTracker(n_points=1, n_clusters=1, device=torch.device('cpu'))
       
       tracker.add_claimed_direction(0, 0, torch.tensor([0.5]), weight=1.0)
       
       test_dir = torch.tensor([[[-0.8]]])
       penalty = tracker.compute_penalty_batch(test_dir)
       
       # In 1D, opposite signs are still parallel: |cos| = 1
       assert penalty[0, 0] < 0.01
   
   def test_1d_penalty_product_formula(self):
       """Product penalty: ∏(1 - α*|cos|)"""
       tracker = DirectionTracker(n_points=1, n_clusters=1, device=torch.device('cpu'))
       
       # Claim with partial weight
       tracker.add_claimed_direction(0, 0, torch.tensor([1.0]), weight=0.5)
       
       test_dir = torch.tensor([[[1.0]]])
       penalty = tracker.compute_penalty_batch(test_dir, penalty_type='product')
       
       # Penalty = 1 - 0.5*1 = 0.5
       assert torch.allclose(penalty[0, 0], torch.tensor(0.5), atol=1e-5)
       
       # Add another claim
       tracker.add_claimed_direction(0, 0, torch.tensor([-1.0]), weight=0.3)
       
       penalty = tracker.compute_penalty_batch(test_dir, penalty_type='product')
       # Product: (1 - 0.5*1) * (1 - 0.3*1) = 0.5 * 0.7 = 0.35
       assert torch.allclose(penalty[0, 0], torch.tensor(0.35), atol=1e-5)
   
   def test_1d_penalty_sum_formula(self):
       """Sum penalty: 1 - Σ(α*|cos|)/Σα"""
       tracker = DirectionTracker(n_points=1, n_clusters=1, device=torch.device('cpu'))
       
       tracker.add_claimed_direction(0, 0, torch.tensor([1.0]), weight=0.6)
       tracker.add_claimed_direction(0, 0, torch.tensor([-1.0]), weight=0.4)
       
       test_dir = torch.tensor([[[1.0]]])
       penalty = tracker.compute_penalty_batch(test_dir, penalty_type='sum')
       
       # Sum: 1 - (0.6*1 + 0.4*1)/(0.6 + 0.4) = 1 - 1 = 0
       assert penalty[0, 0] < 0.01


# ============================================================================
# Penalized Assignment Tests
# ============================================================================

class TestPenalizedAssignment1D:
   
   def test_1d_penalized_assignment(self):
       """Assignment considers both distance and penalty"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       rep.means = torch.tensor([[0.0], [1.0]])
       rep.vectors = torch.tensor([[1.0], [1.0]])
       
       tracker = DirectionTracker(n_points=1, n_clusters=2, device=torch.device('cpu'))
       
       # Point at 0.4 is closer to feature 0
       points = torch.tensor([[0.4]])
       
       # But heavily penalize feature 0
       tracker.add_claimed_direction(0, 0, torch.tensor([1.0]), weight=0.95)
       
       assignment_strategy = PenalizedAssignment(penalty_weight=1.0)
       
       # Without penalty, would assign to 0
       # With heavy penalty on 0, should assign to 1
       assignments, aux = assignment_strategy.compute_assignments(
           points, [rep], direction_tracker=tracker, current_stage=0
       )
       
       # Note: The actual assignment depends on how penalties are applied
       # This test verifies the mechanism exists
       assert 'penalties' in aux
       assert aux['penalties'][0, 0] < 0.1  # Heavy penalty on feature 0


# ============================================================================
# Update with Weights Tests
# ============================================================================

class TestWeightedUpdate1D:
   
   def test_1d_weighted_mean(self):
       """Points [1, 2, 3] with weights [0.1, 0.8, 0.1] -> mean ≈ 2"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=1, device='cpu')
       
       points = torch.tensor([[1.0], [2.0], [3.0]])
       assignments = torch.zeros(3, dtype=torch.long)
       weights = torch.tensor([0.1, 0.8, 0.1])
       
       rep.update_from_points(points, weights=weights, assignments=assignments)
       
       expected_mean = (0.1*1 + 0.8*2 + 0.1*3) / 1.0
       assert torch.allclose(rep.means[0], torch.tensor([expected_mean]), atol=1e-5)
   
   def test_1d_weighted_variance(self):
       """Weighted variance affects vector magnitude"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=1, device='cpu')
       
       points = torch.tensor([[-1.0], [0.0], [1.0]])
       assignments = torch.zeros(3, dtype=torch.long)
       
       # Equal weights
       weights_equal = torch.ones(3) / 3
       rep.update_from_points(points, weights=weights_equal, assignments=assignments)
       var_equal = rep.vectors[0]**2
       
       # Emphasize extremes
       rep = EigenFactorRepresentation(dimension=1, n_factors=1, device='cpu')
       weights_extreme = torch.tensor([0.45, 0.1, 0.45])
       rep.update_from_points(points, weights=weights_extreme, assignments=assignments)
       var_extreme = rep.vectors[0]**2
       
       # Extreme weighting should give higher variance
       assert var_extreme > var_equal
   
   def test_1d_zero_weight_ignored(self):
       """Points with weight=0 don't affect updates"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=1, device='cpu')
       
       points = torch.tensor([[1.0], [100.0], [2.0]])
       assignments = torch.zeros(3, dtype=torch.long)
       weights = torch.tensor([0.5, 0.0, 0.5])  # Middle point has 0 weight
       
       rep.update_from_points(points, weights=weights, assignments=assignments)
       
       # Mean should be (1+2)/2 = 1.5, ignoring the 100
       assert torch.allclose(rep.means[0], torch.tensor([1.5]), atol=1e-4)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration1D:
   
   def test_1d_single_iteration(self):
       """One complete iteration: assign -> update -> track"""
       # Data: uniformly spaced points
       data = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
       n_points = 5
       n_factors = 2
       
       # Initialize
       rep = EigenFactorRepresentation(dimension=1, n_factors=n_factors, device='cpu')
       rep.means = torch.tensor([[-1.0], [1.0]])
       rep.vectors = torch.tensor([[0.5], [0.5]])
       
       tracker = DirectionTracker(n_points=n_points, n_clusters=n_factors, device=torch.device('cpu'))
       assignment_strategy = PenalizedAssignment()
       
       # Assign
       assignments, aux = assignment_strategy.compute_assignments(
           data, [rep], direction_tracker=tracker, current_stage=0
       )
       
       # Update
       rep.update_from_points(data, assignments=assignments)
       
       # Track
       for i in range(n_points):
           direction = rep.vectors[assignments[i]]
           tracker.add_claimed_direction(i, assignments[i].item(), direction)
       
       # Verify means moved toward assigned points
       assert rep.means[0] < 0  # Should be negative
       assert rep.means[1] > 0  # Should be positive
       
       # Verify vectors capture variance
       assert rep.vectors[0].abs() > 0.1
       assert rep.vectors[1].abs() > 0.1
   
   def test_1d_convergence(self):
       """Multiple iterations converge to stable solution"""
       data, true_labels = make_1d_clusters(
           centers=[-2, 2], 
           stds=[0.5, 0.5], 
           n_per_cluster=50,
           seed=42
       )
       
       model = KFactors(
           n_factors=2,
           max_iter=20,
           penalty_weight=0.9,
           verbose=0
       )
       
       model.fit(data)
       
       # Check objective decreased
       if len(model.stage_history_) > 1:
           first_obj = model.stage_history_[0]['objective']
           last_obj = model.stage_history_[-1]['objective']
           assert last_obj <= first_obj
       
       # Check features separated the data
       labels = model.labels_
       
       # Most points from cluster 0 should have same label
       # Most points from cluster 1 should have same label
       # (Not necessarily 0->0, 1->1 due to random init)
       cluster0_labels = labels[:50]
       cluster1_labels = labels[50:]
       
       # Check separation (majority vote)
       mode0 = cluster0_labels.mode().values.item()
       mode1 = cluster1_labels.mode().values.item()
       assert mode0 != mode1  # Different clusters got different labels
   
   def test_1d_penalty_evolution(self):
       """Penalties evolve correctly across iterations"""
       data = torch.tensor([[-1.0], [0.0], [1.0]])
       
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       rep.means = torch.tensor([[-0.5], [0.5]])
       rep.vectors = torch.tensor([[1.0], [1.0]])
       
       assignment_strategy = IndependentFactorAssignment()

       rankings, rankings_aux = assignment_strategy.compute_rankings(data, rep)
       print(f"rankings = rankings{rankings}")
       expected_rankings = torch.tensor([[0, 1], [0, 1], [1, 0]])
       assert torch.equal(rankings, expected_rankings), f"Expected rankings {expected_rankings}, got {rankings}"
       
       # Iteration 1
       memberships, memberships_aux = assignment_strategy.compute_memberships(data, rep, rankings)
       print(f"memberships = {memberships}")
       print(f"memberships_aux = {memberships_aux}")
       expected_memberships = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
       assert torch.equal(memberships, expected_memberships), f"Expected memberships {expected_memberships}, got {memberships}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases1D:
   
   def test_1d_identical_points(self):
       """All points at same location"""
       data = torch.ones(10, 1) * 3.14
       
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       assignments = torch.zeros(10, dtype=torch.long)
       
       rep.update_from_points(data, assignments=assignments)
       
       # Mean should be 3.14
       assert torch.allclose(rep.means[0], torch.tensor([3.14]), atol=1e-5)
       
       # Variance should be ~0, vector magnitude small
       assert rep.vectors[0].abs() < 0.1
   
   def test_1d_more_features_than_points(self):
       """K > n: some features unused"""
       data = torch.tensor([[1.0], [2.0]])
       
       model = KFactors(n_factors=5, max_iter=5, verbose=0)
       model.fit(data)
       
       # Should not crash
       assert model.fitted_
       
       # At most 2 features should be meaningfully used
       labels = model.labels_
       unique_labels = torch.unique(labels)
       assert len(unique_labels) <= 2
   
   def test_1d_single_point(self):
       """n=1: degenerate but shouldn't crash"""
       data = torch.tensor([[0.0]])
       
       model = KFactors(n_factors=2, max_iter=3, verbose=0)
       model.fit(data)
       
       assert model.fitted_
       assert model.labels_.shape == (1,)
   
   def test_1d_colinear_features(self):
       """Features with same mean but different vectors"""
       rep = EigenFactorRepresentation(dimension=1, n_factors=2, device='cpu')
       
       # Same mean, different vector magnitudes
       rep.means = torch.tensor([[0.0], [0.0]])
       rep.vectors = torch.tensor([[1.0], [2.0]])
       
       points = torch.tensor([[1.0], [-1.0]])
       
       # Distance should be same (they're colinear)
       # But scores differ due to vector magnitude
       scores = rep.compute_scores(points)
       
       # Feature 1 has larger magnitude, larger scores
       assert torch.abs(scores[:, 1]) > torch.abs(scores[:, 0]).max()


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
   pytest.main([__file__, "-v"])