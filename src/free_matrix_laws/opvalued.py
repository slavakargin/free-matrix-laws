# src/free_matrix_laws/opvalued.py
from __future__ import annotations
from typing import Sequence
import numpy as np

__all__ = ["covariance_map", "eta"]

def covariance_map(B: np.ndarray, A: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    r"""
    Apply the operator-valued *covariance map*
    \\[
        \eta(B) \;=\; \sum_{i=1}^s A_i\, B\, A_i,
    \\]
    where \\(A_1,\\ldots,A_s\\) are (typically Hermitian) matrices.

    Parameters
    ----------
    B : (n, n) ndarray
        Matrix the map acts on.
    A : sequence of (n, n) ndarrays **or** a single array of shape (s, n, n)
        The list/stack of matrices $A_i$. In many applications the $A_i$ are Hermitian,
        so $A_i B A_i = A_i B A_i^*$. (The usual Kraus form uses $A_i B A_i^*$;
        here Hermitian makes the two coincide.)

    Returns
    -------
    (n, n) ndarray
        The value of $\eta(B)$, with dtype promoted to `np.result_type(B.dtype, A.dtype)`.

    Raises
    ------
    ValueError
        If shapes are inconsistent.

    Notes
    -----
    • Complexity: $\mathcal O(s\,n^3)$.  
    • If each $A_i$ is Hermitian, $\eta$ is self-adjoint w.r.t. the Hilbert--Schmidt inner product.
    • Implementation uses batched matrix multiplies and sums over the stack of $A_i$.

    Examples
    --------
    >>> import numpy as np
    >>> A = [np.eye(2), 2*np.eye(2)]
    >>> B = np.array([[0., 1.],[1., 0.]])
    >>> covariance_map(B, A)
    array([[0., 5.],
           [5., 0.]])
    """
    B = np.asarray(B)
    A_arr = np.asarray(A)

    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square (n,n); got shape {B.shape!r}")
    n = B.shape[0]

    # Accept either (s,n,n) or a single (n,n) as A
    if A_arr.ndim == 2:
        if A_arr.shape != (n, n):
            raise ValueError(f"A has shape {A_arr.shape!r}, expected {(n, n)!r} to match B.")
        A_arr = A_arr[None, ...]  # (1,n,n)
    elif A_arr.ndim == 3:
        if A_arr.shape[1:] != (n, n):
            raise ValueError(f"A stack has shape {A_arr.shape!r}, expected (s,{n},{n}).")
    else:
        raise ValueError("A must be sequence of (n,n) matrices or a (s,n,n) array.")

    # Promote dtype sensibly (keep complex if present)
    dtype = np.result_type(B.dtype, A_arr.dtype)
    B = B.astype(dtype, copy=False)
    A_arr = A_arr.astype(dtype, copy=False)

    # Vectorized: (s,n,n) @ (n,n) -> (s,n,n); then @ (s,n,n) -> (s,n,n); sum over s -> (n,n)
    tmp = A_arr @ B
    out = np.matmul(tmp, A_arr).sum(axis=0)
    return out

# Convenience alias with the familiar symbol name
eta = covariance_map
