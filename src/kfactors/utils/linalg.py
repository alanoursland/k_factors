"""
Linear algebra utilities for the K-Factors algorithms.

Provides numerically stable implementations of common operations like
orthogonalization, eigendecomposition, and matrix decompositions.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor
import warnings


def orthogonalize_basis(basis: Tensor, tol: float = 1e-7) -> Tensor:
    """Orthonormalize a set of basis vectors using QR decomposition.
    
    Args:
        basis: (d, r) matrix where columns are basis vectors
        tol: Tolerance for detecting rank deficiency
        
    Returns:
        (d, r) orthonormal basis
    """
    if basis.shape[1] == 0:
        return basis  # Empty basis
        
    # Use QR decomposition for numerical stability
    Q, R = torch.linalg.qr(basis)
    
    # Check for rank deficiency
    diag_R = torch.diag(R)
    rank = torch.sum(torch.abs(diag_R) > tol).item()
    
    if rank < basis.shape[1]:
        warnings.warn(f"Basis is rank deficient: rank={rank}, requested={basis.shape[1]}")
        # Return only the non-degenerate columns
        return Q[:, :rank]
        
    return Q


def safe_svd(matrix: Tensor, full_matrices: bool = False, 
             max_iter: int = 100) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute SVD with fallback for numerical issues.
    
    Args:
        matrix: Input matrix
        full_matrices: Whether to compute full U and V matrices
        max_iter: Maximum iterations for iterative methods
        
    Returns:
        U, S, Vt from the SVD decomposition
    """
    try:
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    except Exception as e:
        warnings.warn(f"Standard SVD failed: {e}. Trying alternative methods.")
        
        # Try with double precision
        try:
            matrix_f64 = matrix.double()
            U, S, Vt = torch.linalg.svd(matrix_f64, full_matrices=full_matrices)
            return U.float(), S.float(), Vt.float()
        except:
            pass
            
        # Fallback: use eigendecomposition of Gram matrix
        if matrix.shape[0] <= matrix.shape[1]:
            # Use AA^T
            gram = torch.matmul(matrix, matrix.t())
            eigvals, eigvecs = safe_eigh(gram)
            
            # Sort by eigenvalue magnitude
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            U = eigvecs[:, idx]
            
            # Compute singular values
            S = torch.sqrt(torch.clamp(eigvals, min=0))
            
            # Compute V
            Vt = torch.matmul(U.t(), matrix) / (S.unsqueeze(1) + 1e-10)
            
            return U, S, Vt
        else:
            # Use A^TA
            gram = torch.matmul(matrix.t(), matrix)
            eigvals, eigvecs = safe_eigh(gram)
            
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            V = eigvecs[:, idx]
            
            S = torch.sqrt(torch.clamp(eigvals, min=0))
            
            U = torch.matmul(matrix, V) / (S.unsqueeze(0) + 1e-10)
            
            return U, S, V.t()


def safe_eigh(matrix: Tensor, tol: float = 1e-10) -> Tuple[Tensor, Tensor]:
    """Compute eigendecomposition of symmetric matrix with numerical safeguards.
    
    Args:
        matrix: Symmetric matrix
        tol: Tolerance for symmetry check
        
    Returns:
        eigenvalues, eigenvectors
    """
    # Ensure symmetry
    matrix_sym = 0.5 * (matrix + matrix.t())
    
    if torch.max(torch.abs(matrix - matrix_sym)) > tol:
        warnings.warn("Input matrix is not symmetric; symmetrizing.")
        
    try:
        return torch.linalg.eigh(matrix_sym)
    except Exception as e:
        warnings.warn(f"Standard eigendecomposition failed: {e}. Using power iteration.")
        
        # Fallback to power iteration for top eigenvalues
        return power_iteration_eigh(matrix_sym, k=min(10, matrix.shape[0]))


def power_iteration_eigh(matrix: Tensor, k: int, max_iter: int = 100,
                        tol: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Compute top-k eigenvalues/vectors using power iteration.
    
    Args:
        matrix: Symmetric matrix  
        k: Number of eigenvalues to compute
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        (k,) eigenvalues, (n, k) eigenvectors
    """
    n = matrix.shape[0]
    k = min(k, n)
    
    eigenvalues = torch.zeros(k, device=matrix.device)
    eigenvectors = torch.zeros(n, k, device=matrix.device)
    
    remaining_matrix = matrix.clone()
    
    for i in range(k):
        # Random initialization
        v = torch.randn(n, device=matrix.device)
        v = v / torch.norm(v)
        
        for _ in range(max_iter):
            v_new = torch.matmul(remaining_matrix, v)
            eigenvalue = torch.dot(v, v_new)
            v_new = v_new / torch.norm(v_new)
            
            if torch.norm(v_new - v) < tol:
                break
                
            v = v_new
            
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = v
        
        # Deflate matrix
        remaining_matrix = remaining_matrix - eigenvalue * torch.outer(v, v)
        
    return eigenvalues, eigenvectors


def project_to_orthogonal_complement(vectors: Tensor, basis: Tensor) -> Tensor:
    """Project vectors to orthogonal complement of subspace spanned by basis.
    
    Args:
        vectors: (n, d) vectors to project
        basis: (d, r) orthonormal basis of subspace
        
    Returns:
        (n, d) projected vectors
    """
    if basis.shape[1] == 0:
        return vectors  # No subspace to project out
        
    # Project out component in subspace
    coeffs = torch.matmul(vectors, basis)  # (n, r)
    projections = torch.matmul(coeffs, basis.t())  # (n, d)
    
    return vectors - projections


def gram_schmidt(vectors: Tensor, normalize: bool = True) -> Tensor:
    """Gram-Schmidt orthogonalization.
    
    Args:
        vectors: (d, k) matrix of column vectors
        normalize: Whether to normalize to unit length
        
    Returns:
        (d, k) orthogonal (or orthonormal) vectors
    """
    d, k = vectors.shape
    result = torch.zeros_like(vectors)
    
    for i in range(k):
        # Start with original vector
        result[:, i] = vectors[:, i]
        
        # Subtract projections onto previous vectors
        for j in range(i):
            if normalize:
                proj = torch.dot(result[:, i], result[:, j]) * result[:, j]
            else:
                proj = (torch.dot(result[:, i], result[:, j]) / 
                       torch.dot(result[:, j], result[:, j])) * result[:, j]
            result[:, i] = result[:, i] - proj
            
        # Normalize if requested
        if normalize:
            norm = torch.norm(result[:, i])
            if norm > 1e-10:
                result[:, i] = result[:, i] / norm
            else:
                warnings.warn(f"Zero vector encountered in Gram-Schmidt at position {i}")
                
    return result


def matrix_sqrt(matrix: Tensor, method: str = 'eigh') -> Tensor:
    """Compute matrix square root of positive semi-definite matrix.
    
    Args:
        matrix: Positive semi-definite matrix
        method: 'eigh' or 'svd'
        
    Returns:
        Matrix square root such that result @ result.T ≈ matrix
    """
    if method == 'eigh':
        eigvals, eigvecs = safe_eigh(matrix)
        # Clamp negative eigenvalues to zero
        eigvals = torch.clamp(eigvals, min=0)
        return eigvecs @ torch.diag(torch.sqrt(eigvals))
    elif method == 'svd':
        U, S, _ = safe_svd(matrix)
        return U @ torch.diag(torch.sqrt(S))
    else:
        raise ValueError(f"Unknown method: {method}")


def low_rank_approx(matrix: Tensor, rank: int) -> Tuple[Tensor, Tensor]:
    """Compute low-rank approximation using SVD.
    
    Args:
        matrix: Input matrix
        rank: Target rank
        
    Returns:
        U, Sigma such that U @ Sigma @ U.T approximates matrix
    """
    U, S, Vt = safe_svd(matrix, full_matrices=False)
    
    # Keep only top 'rank' components
    rank = min(rank, min(matrix.shape))
    U_r = U[:, :rank]
    S_r = S[:rank]
    
    return U_r, torch.diag(S_r)


def solve_linear_system(A: Tensor, b: Tensor, method: str = 'lstsq') -> Tensor:
    """Solve linear system Ax = b with fallback methods.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
        method: Solution method ('lstsq', 'cholesky', 'qr')
        
    Returns:
        Solution x
    """
    try:
        if method == 'lstsq':
            return torch.linalg.lstsq(A, b).solution
        elif method == 'cholesky':
            # For positive definite systems
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        elif method == 'qr':
            Q, R = torch.linalg.qr(A)
            return torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        warnings.warn(f"Primary solver failed: {e}. Using Moore-Penrose pseudoinverse.")
        return torch.linalg.pinv(A) @ b