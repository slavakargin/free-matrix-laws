from __future__ import annotations
from typing import Sequence
import numpy as np
import numpy.linalg as la

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


# --- utilities ---

def _as_stack(A):
    """Return a stacked array (s,n,n) and a flag telling if input was list/tuple."""
    if isinstance(A, np.ndarray) and A.ndim == 3:
        return A, "stack"
    if isinstance(A, (list, tuple)):
        return np.stack(A, axis=0), "list"
    if isinstance(A, np.ndarray) and A.ndim == 2:
        return A[None, ...], "single"  # single Kraus operator
    raise TypeError("A must be list/tuple of (n,n) arrays, a stacked (s,n,n) array, or a single (n,n) array.")

def _inv_sqrt_psd(M, eps=1e-12):
    """Hermitian inverse square root via eigendecomposition; clips eigenvalues below eps."""
    H = 0.5 * (M + M.conj().T)
    w, V = la.eigh(H)
    w_clip = np.maximum(w, eps)
    D_inv_sqrt = np.diag(w_clip**-0.5)
    return V @ D_inv_sqrt @ V.conj().T

def _TI_TstarI(A_stack):
    """Compute T(I)=sum A A* and T*(I)=sum A* A for Kraus stack A."""
    TI = np.zeros((A_stack.shape[1], A_stack.shape[2]), dtype=complex)
    TS = np.zeros_like(TI)
    for Ai in A_stack:
        TI += Ai @ Ai.conj().T
        TS += Ai.conj().T @ Ai
    return TI, TS

# --- main API ---

def symmetric_sinkhorn_scale(A, eps=1e-12, return_factors=False, preserve_input_type=True):
    r'''
    One **symmetric OSI step**: given a CP map $T(X)=\sum_i A_i X A_i^\ast$,
    form
    $$
      c_1 := \big(T(I)\big)^{-1/2}, \qquad
      c_2 := \big(T^\ast(I)\big)^{-1/2},
    $$
    and return Kraus operators $B_i := c_1 A_i c_2$ for $\mathcal S(T)=S_{c_1,c_2}(T)$.

    Parameters
    ----------
    A : list/tuple of (n,n) arrays, or stacked array (s,n,n), or single (n,n)
        Kraus operators $A_i$.
    eps : float, default 1e-12
        Eigenvalue floor used in the inverse square root of $T(I)$ and $T^\ast(I)$.
    return_factors : bool, default False
        If True, also return $(c_1, c_2)$.
    preserve_input_type : bool, default True
        If True, return a list if input was a list; otherwise return a stacked array.

    Returns
    -------
    B : same container type as A (unless preserve_input_type=False)
        Scaled Kraus operators for $\mathcal S(T)$.
    (optional) c1, c2 : (n,n) arrays
        The scaling factors.
    '''
    A_stack, mode = _as_stack(A)
    s, n, _ = A_stack.shape

    TI, TS = _TI_TstarI(A_stack)
    c1 = _inv_sqrt_psd(TI, eps=eps)
    c2 = _inv_sqrt_psd(TS, eps=eps)

    B_stack = np.empty_like(A_stack, dtype=complex)
    for i in range(s):
        B_stack[i] = c1 @ A_stack[i] @ c2

    if preserve_input_type and mode in {"list", "single"}:
        B_out = [B_stack[i] for i in range(B_stack.shape[0])]
        if mode == "single":
            B_out = B_out[0]
    else:
        B_out = B_stack

    if return_factors:
        return B_out, c1, c2
    return B_out

def symmetric_sinkhorn_apply(X, A, eps=1e-12):
    r'''
    Apply the **symmetric OSI** scaled map $\mathcal S(T)$ to a matrix $X$.

    Given $T(X)=\sum_i A_i X A_i^\ast$ and
    $c_1=(T(I))^{-1/2}$, $c_2=(T^\ast(I))^{-1/2}$,
    this returns
    $$
      \mathcal S(T)(X) \;=\; \sum_i (c_1 A_i c_2)\, X \,(c_1 A_i c_2)^\ast.
    $$

    Parameters
    ----------
    X : (n,n) array
        Input matrix.
    A : list/tuple of (n,n) arrays, or stacked array (s,n,n), or single (n,n)
        Kraus operators $A_i$.
    eps : float, default 1e-12
        Eigenvalue floor in the inverse square roots.

    Returns
    -------
    Y : (n,n) array
        $(\mathcal S(T))(X)$.
    '''
    B_stack, _, _ = symmetric_sinkhorn_scale(A, eps=eps, return_factors=True, preserve_input_type=False)
    Y = np.zeros_like(X, dtype=complex)
    for Bi in B_stack:
        Y += Bi @ X @ Bi.conj().T
    return Y

def symmetric_osi(A, maxiter=50, tol=1e-10, eps=1e-12, return_history=False):
    r'''
    Run **symmetric OSI**: $T,\ \mathcal S(T),\ \mathcal S^2(T),\dots$
    on Kraus operators until (approximately) doubly stochastic:
    $T(I)\approx I$ and $T^\ast(I)\approx I$.

    Parameters
    ----------
    A : list/tuple of (n,n) arrays or stacked (s,n,n)
        Initial Kraus operators.
    maxiter : int
        Maximum iterations.
    tol : float
        Stop when $\|T(I)-I\|_F^2 + \|T^\ast(I)-I\|_F^2 \le \text{tol}$.
    eps : float
        Eigenvalue floor for inverse square roots.
    return_history : bool
        If True, also return a dict with diagnostics.

    Returns
    -------
    B : stacked (s,n,n) array
        Final Kraus operators after scaling.
    info : dict (optional)
        Keys: 'iters', 'ds', 'history' (list of DS distances).
    '''
    B_stack, _ = _as_stack(A)
    hist = []
    for k in range(1, maxiter+1):
        B_stack = symmetric_sinkhorn_scale(B_stack, eps=eps, preserve_input_type=False)
        TI, TS = _TI_TstarI(B_stack)
        n = TI.shape[0]
        I = np.eye(n, dtype=complex)
        ds = float(la.norm(TI - I, 'fro')**2 + la.norm(TS - I, 'fro')**2)
        hist.append(ds)
        if ds <= tol:
            break
    if return_history:
        return B_stack, {"iters": k, "ds": ds, "history": hist}
    return B_stack

def ds_distance(A):
    r'''
    Distance to doubly stochastic class
    $$
      \mathrm{DS}(T) \;=\; \|T(I)-I\|_F^2 \;+\; \|T^\ast(I)-I\|_F^2,
    $$
    for $T(X)=\sum_i A_i X A_i^\ast$ with Kraus $A_i$.
    '''
    A_stack, _ = _as_stack(A)
    TI, TS = _TI_TstarI(A_stack)
    n = TI.shape[0]
    I = np.eye(n, dtype=complex)
    return float(la.norm(TI - I, 'fro')**2 + la.norm(TS - I, 'fro')**2)
