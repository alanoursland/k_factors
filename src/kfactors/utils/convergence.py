"""
Convergence criteria for clustering algorithms.

Different algorithms may use different convergence criteria:
- Change in assignments
- Change in objective function
- Change in parameters
"""

from typing import Dict, Any, Optional
import torch
from torch import Tensor

from ..base.interfaces import ConvergenceCriterion


class ChangeInAssignments(ConvergenceCriterion):
    """Convergence based on fraction of points that change clusters."""
    
    def __init__(self, min_change_fraction: float = 1e-4, 
                 patience: int = 1):
        """
        Args:
            min_change_fraction: Minimum fraction of points that must change
            patience: Number of iterations to wait before declaring convergence
        """
        super().__init__()
        self.min_change_fraction = min_change_fraction
        self.patience = patience
        self._prev_assignments = None
        self._stable_count = 0
        
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Check if assignments have stabilized."""
        assignments = current_state['assignments']
        
        if isinstance(assignments, Tensor):
            current_assignments = assignments
        else:
            # AssignmentMatrix object
            current_assignments = assignments.get_hard()
            
        if self._prev_assignments is None:
            self._prev_assignments = current_assignments.clone()
            return False
            
        # Compute fraction of changed assignments
        n_changed = (current_assignments != self._prev_assignments).sum().item()
        n_total = len(current_assignments)
        change_fraction = n_changed / n_total
        
        # Update history
        self.history.append({
            'iteration': current_state.get('iteration', len(self.history)),
            'n_changed': n_changed,
            'change_fraction': change_fraction
        })
        
        # Check if stable
        if change_fraction < self.min_change_fraction:
            self._stable_count += 1
            converged = self._stable_count >= self.patience
        else:
            self._stable_count = 0
            converged = False
            
        # Update previous assignments
        self._prev_assignments = current_assignments.clone()
        
        return converged
        
        
class ChangeInObjective(ConvergenceCriterion):
    """Convergence based on relative change in objective function."""
    
    def __init__(self, rel_tol: float = 1e-4, abs_tol: float = 1e-8,
                 patience: int = 1):
        """
        Args:
            rel_tol: Relative tolerance for objective change
            abs_tol: Absolute tolerance for objective change  
            patience: Number of iterations to wait before convergence
        """
        super().__init__()
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.patience = patience
        self._prev_objective = None
        self._stable_count = 0
        
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Check if objective has stabilized."""
        current_objective = current_state['objective']
        
        if self._prev_objective is None:
            self._prev_objective = current_objective
            return False
            
        # Compute change
        abs_change = abs(current_objective - self._prev_objective)
        
        if abs(self._prev_objective) > 1e-10:
            rel_change = abs_change / abs(self._prev_objective)
        else:
            rel_change = abs_change
            
        # Update history
        self.history.append({
            'iteration': current_state.get('iteration', len(self.history)),
            'objective': current_objective,
            'abs_change': abs_change,
            'rel_change': rel_change
        })
        
        # Check if stable
        if abs_change < self.abs_tol or rel_change < self.rel_tol:
            self._stable_count += 1
            converged = self._stable_count >= self.patience
        else:
            self._stable_count = 0
            converged = False
            
        # Update previous objective
        self._prev_objective = current_objective
        
        return converged


class ParameterChange(ConvergenceCriterion):
    """Convergence based on change in cluster parameters."""
    
    def __init__(self, tol: float = 1e-6, patience: int = 1,
                 parameter: str = 'mean'):
        """
        Args:
            tol: Tolerance for parameter change (relative Frobenius norm)
            patience: Number of iterations to wait
            parameter: Which parameter to monitor ('mean', 'basis', etc.)
        """
        super().__init__()
        self.tol = tol
        self.patience = patience
        self.parameter = parameter
        self._prev_params = None
        self._stable_count = 0
        
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Check if parameters have stabilized."""
        cluster_state = current_state['cluster_state']
        
        # Extract relevant parameters
        if self.parameter == 'mean':
            current_params = cluster_state.means
        elif self.parameter == 'basis' and cluster_state.bases is not None:
            current_params = cluster_state.bases
        elif self.parameter == 'covariance' and cluster_state.covariances is not None:
            current_params = cluster_state.covariances
        else:
            # Fallback to means
            current_params = cluster_state.means
            
        if self._prev_params is None:
            self._prev_params = current_params.clone()
            return False
            
        # Compute relative change
        diff_norm = torch.norm(current_params - self._prev_params, p='fro')
        prev_norm = torch.norm(self._prev_params, p='fro')
        
        if prev_norm > 1e-10:
            rel_change = (diff_norm / prev_norm).item()
        else:
            rel_change = diff_norm.item()
            
        # Update history
        self.history.append({
            'iteration': current_state.get('iteration', len(self.history)),
            'parameter_change': rel_change
        })
        
        # Check if stable
        if rel_change < self.tol:
            self._stable_count += 1
            converged = self._stable_count >= self.patience
        else:
            self._stable_count = 0
            converged = False
            
        # Update previous parameters
        self._prev_params = current_params.clone()
        
        return converged


class CombinedCriterion(ConvergenceCriterion):
    """Combine multiple convergence criteria with AND/OR logic."""
    
    def __init__(self, criteria: list[ConvergenceCriterion], 
                 mode: str = 'any'):
        """
        Args:
            criteria: List of convergence criteria
            mode: 'any' (OR) or 'all' (AND)
        """
        super().__init__()
        self.criteria = criteria
        self.mode = mode
        
        if mode not in ['any', 'all']:
            raise ValueError(f"Mode must be 'any' or 'all', got {mode}")
            
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Check all criteria and combine results."""
        results = [criterion.check(current_state) for criterion in self.criteria]
        
        if self.mode == 'any':
            converged = any(results)
        else:
            converged = all(results)
            
        # Update history
        self.history.append({
            'iteration': current_state.get('iteration', len(self.history)),
            'individual_results': results,
            'converged': converged
        })
        
        return converged
        
    def reset(self):
        """Reset all sub-criteria."""
        super().reset()
        for criterion in self.criteria:
            criterion.reset()


class MaxIterations(ConvergenceCriterion):
    """Simple convergence after maximum iterations (handled by base class)."""
    
    def check(self, current_state: Dict[str, Any]) -> bool:
        """Never converges - relies on max_iter in base algorithm."""
        return False