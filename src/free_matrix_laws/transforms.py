"""
Numerical transforms for matrix-/operator-valued free probability.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la
from typing import Callable, Optional, Tuple


from .opvalued import covariance_map as eta  # η(B)=Σ A_i B A_i^*

def _hfs_map(G: np.ndarray, z: complex, A) -> np.ndarray:
    r'''
    Half-averaged fixed-point step for the operator-valued semicircle Cauchy transform.

    We seek $G(z)$ solving Speicher's equation
    $$
        z\,G \;=\; I \;+\; \eta(G)\,G, \qquad \Im z>0,
    $$
    where $\eta(B)=\sum_{i=1}^s A_i\,B\,A_i^\ast$ is a completely positive (Kraus) map.

    This iteration map applies the *half-averaged* Picard step
    $$
        T(G)\;=\;\tfrac12\Big( G \;+\; (\,zI - \eta(G)\,)^{-1}\Big),
    $$
    and is often more stable than the raw resolvent update for $\Im z>0$.

    Parameters
    ----------
    G : (n, n) array_like (complex recommended)
        Current iterate for $G(z)$.
    z : complex
        Spectral parameter with $\Im z>0$ (ensures resolvent well-defined).
    A : sequence of (n, n) arrays or stacked array (s, n, n)
        Kraus operators $A_i$ defining $\eta$.

    Returns
    -------
    (n, n) ndarray
        The next iterate $T(G)$.

    Notes
    -----
    • Uses the CP form with $A_i^\ast$ so **no** self-adjointness of $A_i$ is required.  
    • Requires $(zI-\eta(G))$ to be invertible; for $\Im z>0$ this holds in the
      standard operator-valued semicircle setup.

    References
    ----------
    • R. Speicher, *Combinatorial theory of the free product with amalgamation
      and operator-valued free probability theory*, Mem. AMS **132** (627), 1998.  
    • R. Rashidi Far, T. Oraby, W. Bryc, R. Speicher, *Spectra of large block matrices*, 2006.
    '''
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square (n,n); got {G.shape!r}")
    n = G.shape[0]

    # ensure complex path (so conjugations inside η behave as expected)
    dtype = np.result_type(G.dtype, np.complex64)
    I = np.eye(n, dtype=dtype)

    z = np.asarray(z, dtype=np.complex128).item()
    if np.imag(z) <= 0:
        raise ValueError("Require Im(z) > 0 for the semicircle Cauchy transform.")

    K = la.inv(z * I - eta(G.astype(dtype, copy=False), A))
    return 0.5 * (G.astype(dtype, copy=False) + K)


def solve_cauchy_semicircle(z: complex, A, G0: np.ndarray | None = None,
                            tol: float = 1e-10, maxiter: int = 500) -> np.ndarray:
    r'''
    Solve the operator-valued semicircle equation
    $$ z\,G \;=\; I \;+\; \eta(G)\,G, \qquad \Im z>0, $$
    by fixed-point iteration using the half-averaged map
    $$ G \;\mapsto\; \tfrac12\Big[\,G + (\,zI - \eta(G)\,)^{-1}\Big]. $$

    This follows the numerical damping suggested by Helton-Rashidi Far–Speicher (IMRN 2007).

    Parameters
    ----------
    z : complex
        Spectral parameter with $\Im z>0$.
    A : sequence of $(n,n)$ arrays or stacked $(s,n,n)$ array
        Kraus operators $A_i$ defining $\eta(B)=\sum_i A_i B A_i^\ast$.
    G0 : (n,n) array, optional
        Initial iterate (defaults to $-iI$).
    tol : float
        Relative fixed-point tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    (n, n) ndarray
        Approximate solution $G(z)$.

    Notes
    -----
    The residual $R=zG-I-\eta(G)G$ should be small at convergence.
    '''
    # infer n from A
    n = (A[0].shape[0] if isinstance(A, (list, tuple)) else A.shape[-1])
    G = (-1j * np.eye(n)) if G0 is None else np.array(G0, dtype=complex)

    for _ in range(maxiter):
        G_next = _hfs_map(G, z, A)
        if la.norm(G_next - G) <= tol * (1 + la.norm(G)):
            return G_next
        G = G_next
    return G


#public alias
solve_G = solve_cauchy_semicircle

##scalar observables
def semicircle_density(x: float,
                A,
                eps: float = 1e-2,
                G0=None,
                tol: float = 1e-10,
                maxiter: int = 10_000) -> float:
    r'''
    Stieltjes inversion for the **matrix semicircle** at a real point $x$.

    We first compute the operator-valued Cauchy transform $G(z)$ for
    $z = x + i\,\varepsilon$ by solving
    $$
        z\,G \;=\; I \;+\; \eta(G)\,G, \qquad \Im z>0,
    $$
    where $\eta(B)=\sum_{i=1}^s A_i\,B\,A_i^\ast$,
    and then return the scalar density via the normalized trace
    $$
        f(x) \;=\; -\frac{1}{\pi}\,\Im\!\left(\frac{1}{n}\,\mathrm{tr}\,G(x+i\varepsilon)\right).
    $$

    Parameters
    ----------
    x : float
        Real evaluation point.
    A : sequence of $(n,n)$ arrays or stacked array $(s,n,n)$
        Kraus operators $A_i$ (no self-adjointness required).
    eps : float, default 1e-2
        Imaginary offset $\varepsilon>0$. Smaller $\varepsilon$ gives a sharper
        approximation but may need tighter tolerances.
    G0 : (n,n) array, optional
        Initial iterate for $G$ (default $-iI$ inside the solver).
    tol : float, default 1e-10
        Relative fixed-point tolerance for the solver.
    maxiter : int, default 10000
        Maximum iterations.

    Returns
    -------
    float
        Approximation to $f(x)$.

    Notes
    -----
    This computes $m(z)=\tfrac{1}{n}\mathrm{tr}\,G(z)$ and applies the Stieltjes
    formula $f(x)=-(1/\pi)\Im m(x+i\varepsilon)$.  See Helton–Rashidi Far–Speicher
    (IMRN 2007) for the half-averaged fixed-point step used in the solver.
    '''
    if eps <= 0:
        raise ValueError("eps must be > 0")

    # Infer n from A (list/tuple or stacked array)
    if isinstance(A, np.ndarray) and A.ndim == 3:
        n = A.shape[-1]
    elif isinstance(A, np.ndarray) and A.ndim == 2:
        n = A.shape[0]
        A = A[None, ...]
    else:
        n = A[0].shape[0]

    z = float(x) + 1j * float(eps)
    G = solve_cauchy_semicircle(z, A, G0=G0, tol=tol, maxiter=maxiter)
    m = np.trace(G) / n
    f = (-1.0 / np.pi) * np.imag(m)
    return float(f)

#public alias
get_density = semicircle_density


def semicircle_density_scalar(x, c: float = 1.0):
    r'''
    Classical (scalar) Wigner semicircle density with variance $c>0$.

    Support is $[-2\sqrt{c},\,2\sqrt{c}]$ with
    $$
      f(x) \;=\; \frac{1}{2\pi c}\,\sqrt{\,4c - x^2\,}\,\mathbf 1_{\{|x|\le 2\sqrt{c}\}}.
    $$

    Parameters
    ----------
    x : float or array_like
        Evaluation point(s).
    c : float, default 1.0
        Variance parameter ($c>0$).

    Returns
    -------
    float or ndarray
        $f(x)$, vectorized over `x`.

    Notes
    -----
    Parameterization by variance $c$ (so radius is $2\sqrt{c}$).
    '''
    if c <= 0:
        raise ValueError("c must be > 0")
    x_arr = np.asarray(x, dtype=float)
    inside = 4.0 * c - x_arr**2
    y = np.where(inside > 0.0, (1.0 / (2.0 * np.pi * c)) * np.sqrt(inside), 0.0)
    return y if x_arr.ndim else float(y)

