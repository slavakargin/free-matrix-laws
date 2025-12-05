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
def semicircle_density(
    x: float,
    A,
    eps: float = 1e-2,
    G0=None,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    a0=None,
) -> float:
    r'''
    Stieltjes inversion for the **matrix semicircle** (optionally with bias).

    Unbiased case ($a_0$ is `None`): compute $G(z)$ for $z=x+i\varepsilon$ from
    $$ z\,G \;=\; I \;+\; \eta(G)\,G, \qquad \eta(B)=\sum_{i=1}^s A_i B A_i^\ast, $$
    then return the scalar density
    $$ f(x) \;=\; -\frac{1}{\pi}\,\Im\!\left(\frac{1}{n}\,\mathrm{tr}\,G(x+i\varepsilon)\right). $$

    Biased case ($a_0\neq 0$): compute the Cauchy transform $G_{a_0+X}(z)$ via the
    biased solver (internally equivalent to the half-averaged Helton–Rashidi
    Far–Speicher iteration with $b=z(zI-a_0)^{-1}$), and apply the same reduction.

    Parameters
    ----------
    x : float
        Real evaluation point.
    A : sequence of $(n,n)$ arrays or stacked array $(s,n,n)$
        Kraus operators $A_i$ (no self-adjointness required).
    eps : float, default 1e-2
        Imaginary offset $\varepsilon>0$ used for $z=x+i\varepsilon$.
    G0 : (n,n) array, optional
        Initial iterate for $G$ (passed to the solver).
    tol : float, default 1e-10
        Relative fixed-point tolerance for the solver.
    maxiter : int, default 10000
        Maximum iterations.
    a0 : (n,n) array or `None`, default `None`
        Bias matrix. If provided, computes the density for $a_0 + \sum_i A_i \otimes X_i$.

    Returns
    -------
    float
        Approximation to $f(x)$.

    Notes
    -----
    This computes $m(z)=\tfrac{1}{n}\mathrm{tr}\,G(z)$ and uses
    $f(x)=-(1/\pi)\Im m(x+i\varepsilon)$.
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
        A = list(A)
        n = A[0].shape[0]

    if a0 is not None:
        a0 = np.asarray(a0)
        if a0.shape != (n, n):
            raise ValueError(f"a0 must have shape {(n,n)}, got {a0.shape}")

    z = float(x) + 1j * float(eps)

    if a0 is None:
        G = solve_cauchy_semicircle(z, A, G0=G0, tol=tol, maxiter=maxiter)
    else:
        # default return of solve_cauchy_biased is G (not (G, info))
        G = solve_cauchy_biased(z, a0, A, G0=G0, tol=tol, maxiter=maxiter)

    m = np.trace(G) / n
    f = (-1.0 / np.pi) * np.imag(m)
    return float(f)

#public alias
get_density = semicircle_density

def biased_semicircle_density(
    x: float,
    a0,
    A,
    eps: float = 1e-2,
    G0=None,
    tol: float = 1e-10,
    maxiter: int = 10_000,
) -> float:
    r'''
    Convenience alias for the biased case:
    $$ f_{a_0}(x) \;=\; -\frac{1}{\pi}\,\Im\!\left(\frac{1}{n}\,\mathrm{tr}\,G_{a_0+X}(x+i\varepsilon)\right). $$
    Calls `semicircle_density(x, A, eps, G0, tol, maxiter, a0=a0)`.
    '''
    return semicircle_density(x, A, eps=eps, G0=G0, tol=tol, maxiter=maxiter, a0=a0)



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

def hfsb_map(G: np.ndarray,
             z: complex,
             a0: np.ndarray,
             A,
             relax: float = 0.5) -> np.ndarray:
    r'''
    One step of the Helton–Far–Speicher style averaging for the **biased** operator-valued
    semicircle. For $S=a_0 + \sum_i A_i \otimes X_i$ with semicircular $X_i$ and covariance
    $\eta(B)=\sum_i A_i B A_i^\ast$, the Cauchy transform $G(z)$ solves
    $$
      G \;=\; (z I - a_0 - \eta(G))^{-1}
      \quad\Longleftrightarrow\quad
      (z I - a_0)\,G \;=\; I + \eta(G)\,G .
    $$
    Define $b := z\,(z I - a_0)^{-1}$ and the map
    $$
      \Phi(G) \;=\; [\,z I - b\,\eta(G)\,]^{-1} b ,
    $$
    then this routine returns the relaxed step
    $$
      G_{\text{new}} \;=\; (1-\text{relax})\,G \;+\; \text{relax}\,\Phi(G).
    $$

    Parameters
    ----------
    G : (n,n) ndarray
        Current iterate (aiming for $G(z)$ with $\Im z>0$).
    z : complex
        Spectral parameter with $\Im z>0$.
    a0 : (n,n) ndarray
        Bias matrix $a_0$.
    A : sequence[(n,n)] or (s,n,n) ndarray
        Kraus operators defining $\eta$ (list/tuple or stacked array).
    relax : float, default 0.5
        Averaging parameter in $(0,1]$; use $0.5$ for robust damping.

    Returns
    -------
    (n,n) ndarray
    '''
    n = G.shape[0]
    I = np.eye(n, dtype=complex)
    # b = z (z I - a0)^{-1}
    b = z * la.inv(z * I - a0)
    # Φ(G) = [z I - b η(G)]^{-1} b
    Phi = la.inv(z * I - b @ eta(G, A)) @ b
    return (1.0 - relax) * G + relax * Phi


def solve_cauchy_biased(z: complex,
                        a0: np.ndarray,
                        A,
                        G0: np.ndarray | None = None,
                        tol: float = 1e-12,
                        maxiter: int = 5000,
                        relax: float = 0.5,
                        return_info: bool = False):
    r'''
    Fixed-point solver for the **biased** operator-valued semicircle:
    $$
      G \;=\; (z I - a_0 - \eta(G))^{-1},
      \qquad \Im z>0 .
    $$

    Initial guess chosen so that $\Im G(z)<0$ when $\Im z>0$

    Iteration:
    $$
      G_{k+1} \;=\; (1-\text{relax})\,G_k \;+\; \text{relax}\,[\,z I - b\,\eta(G_k)\,]^{-1} b,
      \quad b=z(z I - a_0)^{-1}.
    $$

    Convergence check uses the equation form
    $$
      R(G):= (z I - a_0)\,G - I - \eta(G)\,G
    $$
    and stops when $\|R(G)\|_{\mathrm{F}} \le \text{tol}$.

    Parameters
    ----------
    z : complex
        Spectral parameter with $\Im z>0$.
    a0 : (n,n) ndarray
        Bias matrix.
    A : sequence[(n,n)] or (s,n,n) ndarray
        Kraus operators for $\eta$.
    G0 : (n,n) ndarray, optional
        Warm start; if None, uses $(\Im z)^{-1} i\,I$.
    tol : float, default 1e-12
        Frobenius-norm tolerance on the residual.
    maxiter : int, default 5000
        Iteration cap.
    relax : float, default 0.5
        Averaging parameter in $(0,1]$.
    return_info : bool, default False
        If True, also return a dict with residual and iterations.

    Returns
    -------
    G : (n,n) ndarray
    info : dict (only if return_info=True)
        Keys: ``residual``, ``iters``.
    '''
    n = a0.shape[0]
    I = np.eye(n, dtype=complex)
    if G0 is None:
        eps = float(np.imag(z))
        if eps <= 0:
            eps = 1e-2
        #(Herglotz sign: Im G(z) < 0 for Im z > 0)
        G = -1j * I / eps
    else:
        G = G0.astype(complex, copy=True)

    for k in range(1, maxiter + 1):
        G = hfsb_map(G, z, a0, A, relax=relax)
        # residual: (zI - a0)G - I - eta(G)G
        r = (z * I - a0) @ G - I - eta(G, A) @ G
        res = la.norm(r, 'fro')
        if res <= tol:
            if return_info:
                return G, {"residual": res, "iters": k}
            return G
    if return_info:
        return G, {"residual": res, "iters": maxiter}
    return G
