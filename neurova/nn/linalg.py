# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Linear Algebra Module.

Neurova implementation of linear algebra operations for tensor computations
in deep learning and scientific computing.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, List
from scipy import linalg as scipy_linalg


# Matrix Decompositions

def cholesky(A: np.ndarray, upper: bool = False, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Cholesky decomposition of a positive-definite matrix.
    
    Args:
        A: Input positive-definite matrix (..., M, M)
        upper: If True, return upper triangular matrix
        out: Optional output array
        
    Returns:
        Lower (or upper if upper=True) triangular Cholesky factor
    """
    if A.ndim == 2:
        L = np.linalg.cholesky(A)
        result = L.T if upper else L
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        result = np.zeros(A.shape)
        for idx in np.ndindex(batch_shape):
            L = np.linalg.cholesky(A[idx])
            result[idx] = L.T if upper else L
    
    if out is not None:
        out[:] = result
        return out
    return result


def cholesky_ex(A: np.ndarray, upper: bool = False, check_errors: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Cholesky decomposition with error info.
    
    Returns:
        Tuple of (factor, info) where info > 0 indicates failure at that position
    """
    try:
        L = cholesky(A, upper=upper)
        info = np.zeros(A.shape[:-2] or (1,), dtype=np.int32)
        return L, info
    except np.linalg.LinAlgError as e:
        info = np.ones(A.shape[:-2] or (1,), dtype=np.int32)
        L = np.zeros_like(A)
        return L, info


def eig(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a square matrix.
    
    Args:
        A: Input square matrix (..., M, M)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if A.ndim == 2:
        return np.linalg.eig(A)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        eigenvalues = np.zeros(batch_shape + (M,), dtype=complex)
        eigenvectors = np.zeros(batch_shape + (M, M), dtype=complex)
        for idx in np.ndindex(batch_shape):
            w, v = np.linalg.eig(A[idx])
            eigenvalues[idx] = w
            eigenvectors[idx] = v
        return eigenvalues, eigenvectors


def eigh(A: np.ndarray, UPLO: str = 'L') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix.
    
    Args:
        A: Input Hermitian matrix (..., M, M)
        UPLO: 'L' or 'U' specifying which triangle to use
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) in ascending order
    """
    if A.ndim == 2:
        return np.linalg.eigh(A, UPLO=UPLO)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        eigenvalues = np.zeros(batch_shape + (M,))
        eigenvectors = np.zeros(batch_shape + (M, M))
        for idx in np.ndindex(batch_shape):
            w, v = np.linalg.eigh(A[idx], UPLO=UPLO)
            eigenvalues[idx] = w
            eigenvectors[idx] = v
        return eigenvalues, eigenvectors


def eigvals(A: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of a square matrix."""
    if A.ndim == 2:
        return np.linalg.eigvals(A)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        result = np.zeros(batch_shape + (M,), dtype=complex)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.eigvals(A[idx])
        return result


def eigvalsh(A: np.ndarray, UPLO: str = 'L') -> np.ndarray:
    """Compute eigenvalues of a Hermitian/symmetric matrix."""
    if A.ndim == 2:
        return np.linalg.eigvalsh(A, UPLO=UPLO)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        result = np.zeros(batch_shape + (M,))
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.eigvalsh(A[idx], UPLO=UPLO)
        return result


def qr(A: np.ndarray, mode: str = 'reduced') -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute QR decomposition.
    
    Args:
        A: Input matrix (..., M, N)
        mode: 'reduced', 'complete', or 'r'
        
    Returns:
        If mode != 'r': Tuple of (Q, R)
        If mode == 'r': R only
    """
    if A.ndim == 2:
        return np.linalg.qr(A, mode=mode)
    else:
        batch_shape = A.shape[:-2]
        M, N = A.shape[-2:]
        K = min(M, N)
        
        if mode == 'reduced':
            Q = np.zeros(batch_shape + (M, K))
            R = np.zeros(batch_shape + (K, N))
        elif mode == 'complete':
            Q = np.zeros(batch_shape + (M, M))
            R = np.zeros(batch_shape + (M, N))
        else:  # mode == 'r'
            R = np.zeros(batch_shape + (K, N))
        
        for idx in np.ndindex(batch_shape):
            result = np.linalg.qr(A[idx], mode=mode)
            if mode == 'r':
                R[idx] = result
            else:
                Q[idx], R[idx] = result
        
        if mode == 'r':
            return R
        return Q, R


def svd(A: np.ndarray, full_matrices: bool = True, compute_uv: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute Singular Value Decomposition.
    
    Args:
        A: Input matrix (..., M, N)
        full_matrices: If True, compute full U and Vh
        compute_uv: If False, only compute singular values
        
    Returns:
        If compute_uv: Tuple of (U, S, Vh)
        Else: S only
    """
    if A.ndim == 2:
        return np.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
    else:
        batch_shape = A.shape[:-2]
        M, N = A.shape[-2:]
        K = min(M, N)
        
        S = np.zeros(batch_shape + (K,))
        if compute_uv:
            if full_matrices:
                U = np.zeros(batch_shape + (M, M))
                Vh = np.zeros(batch_shape + (N, N))
            else:
                U = np.zeros(batch_shape + (M, K))
                Vh = np.zeros(batch_shape + (K, N))
        
        for idx in np.ndindex(batch_shape):
            result = np.linalg.svd(A[idx], full_matrices=full_matrices, compute_uv=compute_uv)
            if compute_uv:
                U[idx], S[idx], Vh[idx] = result
            else:
                S[idx] = result
        
        if compute_uv:
            return U, S, Vh
        return S


def svdvals(A: np.ndarray) -> np.ndarray:
    """Compute singular values of a matrix."""
    return svd(A, compute_uv=False)


def lu(A: np.ndarray, pivot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LU decomposition with partial pivoting.
    
    Args:
        A: Input matrix (..., M, N)
        pivot: Whether to use pivoting
        
    Returns:
        Tuple of (P, L, U) permutation, lower, upper triangular
    """
    if A.ndim == 2:
        P, L, U = scipy_linalg.lu(A)
        return P, L, U
    else:
        batch_shape = A.shape[:-2]
        M, N = A.shape[-2:]
        K = min(M, N)
        
        P = np.zeros(batch_shape + (M, M))
        L = np.zeros(batch_shape + (M, K))
        U = np.zeros(batch_shape + (K, N))
        
        for idx in np.ndindex(batch_shape):
            P[idx], L[idx], U[idx] = scipy_linalg.lu(A[idx])
        
        return P, L, U


def lu_factor(A: np.ndarray, pivot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LU factorization for solving linear systems.
    
    Returns:
        Tuple of (LU, pivots)
    """
    if A.ndim == 2:
        return scipy_linalg.lu_factor(A)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        
        LU_list = []
        piv_list = []
        
        for idx in np.ndindex(batch_shape):
            lu, piv = scipy_linalg.lu_factor(A[idx])
            LU_list.append(lu)
            piv_list.append(piv)
        
        LU = np.stack(LU_list).reshape(batch_shape + (M, M))
        pivots = np.stack(piv_list).reshape(batch_shape + (M,))
        
        return LU, pivots


def schur(A: np.ndarray, output: str = 'real') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Schur decomposition.
    
    Args:
        A: Input square matrix
        output: 'real' or 'complex'
        
    Returns:
        Tuple of (T, Z) where A = Z @ T @ Z.H
    """
    return scipy_linalg.schur(A, output=output)


def polar(A: np.ndarray, side: str = 'right') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute polar decomposition A = U @ P (right) or A = P @ U (left).
    
    Args:
        A: Input matrix
        side: 'right' or 'left'
        
    Returns:
        Tuple of (U, P) unitary and positive semi-definite matrices
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    if side == 'right':
        unitary = U @ Vh
        positive = Vh.conj().T @ np.diag(S) @ Vh
    else:
        unitary = U @ Vh
        positive = U @ np.diag(S) @ U.conj().T
    return unitary, positive


# Matrix Properties

def det(A: np.ndarray) -> np.ndarray:
    """Compute determinant of a square matrix."""
    if A.ndim == 2:
        return np.linalg.det(A)
    else:
        batch_shape = A.shape[:-2]
        result = np.zeros(batch_shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.det(A[idx])
        return result


def slogdet(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sign and log of absolute determinant.
    
    Returns:
        Tuple of (sign, logabsdet)
    """
    if A.ndim == 2:
        return np.linalg.slogdet(A)
    else:
        batch_shape = A.shape[:-2]
        sign = np.zeros(batch_shape)
        logabsdet = np.zeros(batch_shape)
        for idx in np.ndindex(batch_shape):
            s, l = np.linalg.slogdet(A[idx])
            sign[idx] = s
            logabsdet[idx] = l
        return sign, logabsdet


def matrix_rank(A: np.ndarray, tol: Optional[float] = None, hermitian: bool = False) -> np.ndarray:
    """Compute matrix rank."""
    if A.ndim == 2:
        return np.linalg.matrix_rank(A, tol=tol, hermitian=hermitian)
    else:
        batch_shape = A.shape[:-2]
        result = np.zeros(batch_shape, dtype=np.int64)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.matrix_rank(A[idx], tol=tol, hermitian=hermitian)
        return result


def cond(A: np.ndarray, p: Optional[Union[int, float, str]] = None) -> np.ndarray:
    """
    Compute condition number.
    
    Args:
        A: Input matrix
        p: Order of the norm (None, 1, 2, inf, -inf, 'fro')
    """
    if A.ndim == 2:
        return np.linalg.cond(A, p=p)
    else:
        batch_shape = A.shape[:-2]
        result = np.zeros(batch_shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.cond(A[idx], p=p)
        return result


def norm(A: np.ndarray, ord: Optional[Union[int, float, str]] = None, 
         dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> np.ndarray:
    """
    Compute matrix or vector norm.
    
    Args:
        A: Input tensor
        ord: Order of norm ('fro', 'nuc', inf, -inf, 1, 2, etc.)
        dim: Dimensions to reduce over
        keepdim: Keep reduced dimensions
    """
    if dim is None:
        if A.ndim <= 2:
            result = np.linalg.norm(A, ord=ord)
        else:
            result = np.linalg.norm(A.reshape(A.shape[:-2] + (-1,)), ord=ord, axis=-1)
    else:
        result = np.linalg.norm(A, ord=ord, axis=dim, keepdims=keepdim)
    return result


def matrix_norm(A: np.ndarray, ord: str = 'fro', dim: Tuple[int, int] = (-2, -1),
                keepdim: bool = False) -> np.ndarray:
    """Compute matrix norm over specified dimensions."""
    if ord == 'fro':
        result = np.sqrt(np.sum(A ** 2, axis=dim, keepdims=keepdim))
    elif ord == 'nuc':
        result = np.sum(svdvals(A), axis=-1, keepdims=keepdim)
    elif ord == 1:
        result = np.max(np.sum(np.abs(A), axis=dim[0], keepdims=True), axis=dim[1], keepdims=keepdim)
    elif ord == -1:
        result = np.min(np.sum(np.abs(A), axis=dim[0], keepdims=True), axis=dim[1], keepdims=keepdim)
    elif ord == 2:
        result = np.max(svdvals(A), axis=-1, keepdims=keepdim)
    elif ord == -2:
        result = np.min(svdvals(A), axis=-1, keepdims=keepdim)
    elif ord == float('inf'):
        result = np.max(np.sum(np.abs(A), axis=dim[1], keepdims=True), axis=dim[0], keepdims=keepdim)
    elif ord == float('-inf'):
        result = np.min(np.sum(np.abs(A), axis=dim[1], keepdims=True), axis=dim[0], keepdims=keepdim)
    else:
        raise ValueError(f"Unknown norm order: {ord}")
    return result


def vector_norm(x: np.ndarray, ord: Union[int, float] = 2, dim: Optional[Union[int, Tuple[int, ...]]] = None,
                keepdim: bool = False) -> np.ndarray:
    """Compute vector norm."""
    if dim is None:
        return np.linalg.norm(x.ravel(), ord=ord)
    return np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim)


# Matrix Inverse and Solve

def inv(A: np.ndarray) -> np.ndarray:
    """Compute matrix inverse."""
    if A.ndim == 2:
        return np.linalg.inv(A)
    else:
        batch_shape = A.shape[:-2]
        M = A.shape[-1]
        result = np.zeros(A.shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.inv(A[idx])
        return result


def pinv(A: np.ndarray, rcond: float = 1e-15, hermitian: bool = False) -> np.ndarray:
    """
    Compute Moore-Penrose pseudoinverse.
    
    Args:
        A: Input matrix
        rcond: Cutoff for small singular values
        hermitian: Assume matrix is Hermitian
    """
    if A.ndim == 2:
        return np.linalg.pinv(A, rcond=rcond, hermitian=hermitian)
    else:
        batch_shape = A.shape[:-2]
        M, N = A.shape[-2:]
        result = np.zeros(batch_shape + (N, M))
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.pinv(A[idx], rcond=rcond, hermitian=hermitian)
        return result


def solve(A: np.ndarray, B: np.ndarray, left: bool = True) -> np.ndarray:
    """
    Solve linear system AX = B (left=True) or XA = B (left=False).
    
    Args:
        A: Coefficient matrix (..., M, M)
        B: Right-hand side (..., M, K) or (..., K, M)
        left: Solve from left (default) or right
    """
    if not left:
        return np.linalg.solve(A.T, B.T).T
    if A.ndim == 2 and B.ndim <= 2:
        return np.linalg.solve(A, B)
    else:
        batch_shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2] if B.ndim > 1 else ())
        result_shape = batch_shape + B.shape[-2:] if B.ndim > 1 else batch_shape + B.shape[-1:]
        result = np.zeros(result_shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.solve(A[idx], B[idx])
        return result


def solve_triangular(A: np.ndarray, B: np.ndarray, upper: bool = True, left: bool = True,
                     unitriangular: bool = False) -> np.ndarray:
    """
    Solve triangular linear system.
    
    Args:
        A: Triangular coefficient matrix
        B: Right-hand side
        upper: A is upper triangular
        left: Solve from left
        unitriangular: Assume diagonal is 1
    """
    if A.ndim == 2 and B.ndim <= 2:
        return scipy_linalg.solve_triangular(A, B, lower=not upper, unit_diagonal=unitriangular)
    else:
        batch_shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2] if B.ndim > 1 else ())
        result_shape = batch_shape + B.shape[-2:] if B.ndim > 1 else batch_shape + B.shape[-1:]
        result = np.zeros(result_shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = scipy_linalg.solve_triangular(A[idx], B[idx], lower=not upper, unit_diagonal=unitriangular)
        return result


def lstsq(A: np.ndarray, B: np.ndarray, rcond: Optional[float] = None, driver: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute least-squares solution.
    
    Args:
        A: Coefficient matrix (..., M, N)
        B: Right-hand side (..., M, K)
        rcond: Cutoff for small singular values
        
    Returns:
        Tuple of (solution, residuals, rank, singular_values)
    """
    if A.ndim == 2 and B.ndim <= 2:
        solution, residuals, rank, s = np.linalg.lstsq(A, B, rcond=rcond)
        return solution, residuals if len(residuals) > 0 else np.array([]), rank, s
    else:
        batch_shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2] if B.ndim > 1 else ())
        N = A.shape[-1]
        K = B.shape[-1] if B.ndim > 1 else 1
        
        solutions = np.zeros(batch_shape + (N, K) if B.ndim > 1 else batch_shape + (N,))
        for idx in np.ndindex(batch_shape):
            sol, _, _, _ = np.linalg.lstsq(A[idx], B[idx], rcond=rcond)
            solutions[idx] = sol
        
        return solutions, np.array([]), np.array([]), np.array([])


# Matrix Products and Operations

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication."""
    return np.matmul(A, B)


def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
    """Raise square matrix to integer power."""
    if A.ndim == 2:
        return np.linalg.matrix_power(A, n)
    else:
        batch_shape = A.shape[:-2]
        result = np.zeros(A.shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = np.linalg.matrix_power(A[idx], n)
        return result


def matrix_exp(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential."""
    if A.ndim == 2:
        return scipy_linalg.expm(A)
    else:
        batch_shape = A.shape[:-2]
        result = np.zeros(A.shape)
        for idx in np.ndindex(batch_shape):
            result[idx] = scipy_linalg.expm(A[idx])
        return result


def cross(a: np.ndarray, b: np.ndarray, dim: int = -1) -> np.ndarray:
    """Compute cross product."""
    return np.cross(a, b, axis=dim)


def diagonal(A: np.ndarray, offset: int = 0, dim1: int = -2, dim2: int = -1) -> np.ndarray:
    """Extract or create diagonal matrix."""
    return np.diagonal(A, offset=offset, axis1=dim1, axis2=dim2)


def trace(A: np.ndarray, offset: int = 0, dim1: int = -2, dim2: int = -1) -> np.ndarray:
    """Compute matrix trace."""
    return np.trace(A, offset=offset, axis1=dim1, axis2=dim2)


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute outer product."""
    return np.outer(a, b)


def inner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute inner product."""
    return np.inner(a, b)


def tensordot(a: np.ndarray, b: np.ndarray, dims: Union[int, Tuple[List[int], List[int]]] = 2) -> np.ndarray:
    """Compute tensor contraction."""
    return np.tensordot(a, b, axes=dims)


def einsum(subscripts: str, *operands: np.ndarray, optimize: bool = True) -> np.ndarray:
    """Einstein summation convention."""
    return np.einsum(subscripts, *operands, optimize=optimize)


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Kronecker product."""
    return np.kron(a, b)


def vander(x: np.ndarray, N: Optional[int] = None, increasing: bool = False) -> np.ndarray:
    """
    Generate Vandermonde matrix.
    
    Args:
        x: Input 1D array
        N: Number of columns (default: len(x))
        increasing: Order of powers
    """
    if N is None:
        N = len(x)
    if increasing:
        return np.vander(x, N=N)[:, ::-1]
    return np.vander(x, N=N)


def householder_product(A: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Compute product of Householder reflectors.
    
    Computes the matrix Q from Householder vectors stored in A.
    """
    M, N = A.shape
    K = len(tau)
    Q = np.eye(M)
    for i in range(K - 1, -1, -1):
        v = np.zeros(M)
        v[i] = 1
        v[i + 1:] = A[i + 1:, i]
        Q = Q - tau[i] * np.outer(v, v @ Q)
    return Q


# Specialized Decompositions

def ldl(A: np.ndarray, hermitian: bool = True, upper: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LDL decomposition of a symmetric/Hermitian matrix.
    
    Returns:
        Tuple of (LD, pivots) where L and D are packed in LD
    """
    L, D, perm = scipy_linalg.ldl(A, lower=not upper, hermitian=hermitian)
    return L @ np.diag(D.diagonal()), perm


def cholesky_solve(B: np.ndarray, L: np.ndarray, upper: bool = False) -> np.ndarray:
    """
    Solve linear system given Cholesky factor.
    
    Solves A @ X = B where A = L @ L.T (or U.T @ U if upper)
    """
    if upper:
        return scipy_linalg.cho_solve((L, True), B)
    return scipy_linalg.cho_solve((L, False), B)


def lu_solve(LU_pivots: Tuple[np.ndarray, np.ndarray], B: np.ndarray) -> np.ndarray:
    """Solve linear system from LU factorization."""
    LU, pivots = LU_pivots
    return scipy_linalg.lu_solve((LU, pivots), B)


# Export all functions
__all__ = [
    # Decompositions
    'cholesky', 'cholesky_ex', 'eig', 'eigh', 'eigvals', 'eigvalsh',
    'qr', 'svd', 'svdvals', 'lu', 'lu_factor', 'schur', 'polar', 'ldl',
    # Properties
    'det', 'slogdet', 'matrix_rank', 'cond', 'norm', 'matrix_norm', 'vector_norm',
    # Inverse and solve
    'inv', 'pinv', 'solve', 'solve_triangular', 'lstsq', 'cholesky_solve', 'lu_solve',
    # Products and operations
    'matmul', 'matrix_power', 'matrix_exp', 'cross', 'diagonal', 'trace',
    'outer', 'inner', 'tensordot', 'einsum', 'kron', 'vander', 'householder_product',
]
