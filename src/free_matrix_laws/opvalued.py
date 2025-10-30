from __future__ import annotations
from typing import Sequence
import numpy as np

__all__ = ["covariance_map", "eta"]

def covariance_map(B: np.ndarray, A: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    r'''
    Apply the completely positive (Kraus) map
    $\eta(B) = \sum_{i=1}^s A_i\, B\, A_i^{\ast}$.

    Parameters
    ----------
    B : (n, n) ndarray
        Square matrix the map acts on.
    A : sequence of (n, n) ndarrays or (s, n, n) ndarray
        Kraus operators $A_i$ (no self-adjointness assumed).

    Returns
    -------
    (n, n) ndarray
        The value of $\eta(B)$.

    Notes
    -----
    • Uses $A_i^{\ast}$ (conjugate transpose), not $A_i^{\mathsf T}$.  
    • Accepts a list/tuple of $(n,n)$ or a stacked array $(s,n,n)$.
    '''
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square (n,n); got {B.shape!r}")
    n = B.shape[0]

    # Normalize A to a stacked array (s,n,n)
    if isinstance(A, np.ndarray) and A.ndim == 3:
        A_arr = A
    elif isinstance(A, np.ndarray) and A.ndim == 2:
        if A.shape != (n, n):
            raise ValueError(f"A has shape {A.shape!r}, expected {(n, n)!r}.")
        A_arr = A[None, ...]
    else:
        # Python sequence → stack
        A_list = list(A)  # type: ignore[arg-type]
        A_arr = np.stack(A_list, axis=0)

    if A_arr.shape[1:] != (n, n):
        raise ValueError(f"A stack has shape {A_arr.shape!r}, expected (s,{n},{n}).")

    # Promote dtype; ensure complex paths keep conjugation correct
    dtype = np.result_type(B.dtype, A_arr.dtype, np.complex64)
    B = B.astype(dtype, copy=False)
    A_arr = A_arr.astype(dtype, copy=False)

    # Vectorized Kraus: sum_i A_i B A_i^*
    A_dag = A_arr.conj().swapaxes(-1, -2)   # (s,n,n)
    tmp = A_arr @ B                         # (s,n,n)
    out = np.matmul(tmp, A_dag).sum(axis=0) # (n,n)
    return out

# Public alias
eta = covariance_map
