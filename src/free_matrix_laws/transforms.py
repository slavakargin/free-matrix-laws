"""
Numerical transforms for matrix-/operator-valued free probability.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la
from numpy.polynomial.legendre import leggauss
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

def semicircle_cauchy_scalar(z, c: float = 1.0):
    r"""
    Scalar Cauchy (Stieltjes) transform of the Wigner semicircle law with variance c>0.

    For c=1 (support [-2,2]):
        G(z) = (z - sqrt(z^2 - 4))/2

    More generally (support [-2*sqrt(c), 2*sqrt(c)]):
        G(z) = (z - sqrt(z^2 - 4c)) / (2c)

    The square-root branch is chosen so that Im(z)>0 => Im(G(z))<0.
    For boundary values on the real line, use z = x + 1j*eps with eps>0.
    """
    if c <= 0:
        raise ValueError("c must be > 0")

    z_arr = np.asarray(z, dtype=np.complex128)
    disc = np.sqrt(z_arr**2 - 4.0 * c)

    # Enforce Herglotz symmetry: Im(z)>0 -> Im(G)<0 (and vice versa)
    disc = np.where(disc.imag * z_arr.imag < 0, -disc, disc)

    G = (z_arr - disc) / (2.0 * c)
    return G if z_arr.ndim else complex(G)


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



def _lambda_eps(z: complex, n: int, eps_reg: float, block_size: int = 1) -> np.ndarray:
    r'''
    Build the regularized linearization parameter
    $$
      \Lambda_\varepsilon(z)
      =
      \begin{bmatrix}
        z I_{k} & 0 \\
        0 & i\varepsilon\, I_{n-k}
      \end{bmatrix},
    $$
    where $k=\text{block\_size}$.

    Notes
    -----
    This is the standard regularization used when extracting the $(1,1)$ corner
    (or more generally the top-left $k\times k$ corner) from the resolvent of
    a self-adjoint linearization.
    '''
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (1 <= block_size <= n):
        raise ValueError(f"block_size must be in {{1,...,n}}; got {block_size}.")
    if eps_reg <= 0:
        raise ValueError("eps_reg must be > 0.")

    Lam = (1j * float(eps_reg)) * np.eye(n, dtype=complex)
    Lam[:block_size, :block_size] = complex(z) * np.eye(block_size, dtype=complex)
    return Lam


def _hfsc_map(
    G: np.ndarray,
    z: complex,
    a0: np.ndarray,
    A,
    eps_reg: float,
    block_size: int = 1,
) -> np.ndarray:
    r'''
    One HFSC half-averaged iteration step for a linearized polynomial problem.

    Given a self-adjoint linearization
    $$
      L_p = a_0 + \sum_{i=1}^s A_i \otimes X_i
    $$
    (with $X_i$ semicircular), define
    $$
      b_\varepsilon(z) := z\big(\Lambda_\varepsilon(z)-a_0\big)^{-1},
      \qquad
      \eta(B) := \sum_{i=1}^s A_i\,B\,A_i^\ast.
    $$
    The HFSC step is
    $$
      G \mapsto \frac12\Big[G + W\Big],
      \qquad
      W := \big(zI - b_\varepsilon(z)\,\eta(G)\big)^{-1} b_\varepsilon(z).
    $$

    Notes
    -----
    This is a low-level routine. Most users should call
    `solve_cauchy_linearized` or `polynomial_density`.
    '''
    G = np.asarray(G)
    a0 = np.asarray(a0)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square; got {G.shape!r}")
    if a0.shape != G.shape:
        raise ValueError(f"a0 must have shape {G.shape!r}; got {a0.shape!r}")

    n = G.shape[0]
    I = np.eye(n, dtype=complex)

    Lam = _lambda_eps(z, n, eps_reg=eps_reg, block_size=block_size)

    # b = z * (Lam - a0)^{-1}
    b = complex(z) * la.solve(Lam - a0, I)

    # W = (z I - b eta(G))^{-1} b   (use solve instead of inv)
    M = complex(z) * I - b @ eta(G, A)
    W = la.solve(M, b)

    return 0.5 * (G + W)


def solve_cauchy_linearized(
    z: complex,
    a0: np.ndarray,
    A,
    *,
    eps_reg: float = 1e-6,
    block_size: int = 1,
    G0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    return_info: bool = False,
):
    r'''
    Solve for the regularized “quasi-resolvent” fixed point associated to a
    self-adjoint linearization $L_p$.

    We assume a self-adjoint linearization
    $$
      L_p = a_0 + \sum_{i=1}^s A_i \otimes X_i,
    $$
    with semicircular $X_i$, and we form
    $$
      b_\varepsilon(z) := z\big(\Lambda_\varepsilon(z)-a_0\big)^{-1},
      \qquad
      \eta(B) := \sum_{i=1}^s A_i\,B\,A_i^\ast,
    $$
    where
    $$
    \Lambda_\varepsilon(z)=\operatorname{diag} \big(z I_k,\ i\varepsilon\, I_{n-k}\big), \qquad k=\text{block\_size}.
    $$

    The iteration uses the half-averaged update
    $$
      G_{new} = \frac12\Big[G + \big(zI - b_\varepsilon(z)\,\eta(G)\big)^{-1} b_\varepsilon(z)\Big],
    $$
    and stops when $\|G_{new}-G\|_F \le \text{tol}\,\|G\|_F$ (relative Frobenius criterion).

    Parameters
    ----------
    z : complex
        Spectral parameter with $\Im z>0$ (typically $z=x+i\,\varepsilon$).
    a0 : (n,n) array
        The bias / constant term of the linearization.
    A : sequence of (n,n) arrays or stacked array (s,n,n)
        Coefficients $A_i$ defining $\eta(B)=\sum_i A_i B A_i^\ast$.
    eps_reg : float, default 1e-6
        Regularization parameter in $\Lambda_\varepsilon(z)$ (used on the lower block).
    block_size : int, default 1
        Size $k$ of the distinguished top-left block (the one used to recover the scalar Cauchy transform).
    G0 : (n,n) array, optional
        Initial iterate. If None, uses $G_0 = (1/z)I$.
    tol : float, default 1e-10
        Relative tolerance.
    maxiter : int, default 10000
        Maximum number of iterations.
    return_info : bool, default False
        If True, also return a dict with diagnostics.

    Returns
    -------
    G : (n,n) array
        Approximation to the fixed point $G(z,b_\varepsilon(z))$.
    info : dict (optional)
        Keys: 'iters', 'last_diff'.

    Notes
    -----
    After computing $G(z,b_\varepsilon(z))$, the scalar Cauchy transform of $p$
    is obtained from the distinguished corner via
    $$
      m_p(z) \approx \frac{1}{k}\,\mathrm{tr}\,\big(G(z,b_\varepsilon(z))\big)_{11},
    $$
    with $k=\text{block\_size}$ and $(\cdot)_{11}$ the top-left $k\times k$ block.
    '''
    a0 = np.asarray(a0)
    if a0.ndim != 2 or a0.shape[0] != a0.shape[1]:
        raise ValueError(f"a0 must be square; got {a0.shape!r}")
    n = a0.shape[0]

    if G0 is None:
        G = (1.0 / complex(z)) * np.eye(n, dtype=complex)
    else:
        G = np.asarray(G0, dtype=complex)
        if G.shape != (n, n):
            raise ValueError(f"G0 must have shape {(n,n)!r}; got {G.shape!r}")

    last_diff = np.inf
    for k in range(1, maxiter + 1):
        G1 = _hfsc_map(G, z, a0, A, eps_reg=eps_reg, block_size=block_size)
        diff = la.norm(G1 - G, "fro")
        denom = max(1.0, la.norm(G, "fro"))
        last_diff = diff

        if diff <= tol * denom:
            G = G1
            break

        G = G1

    if return_info:
        return G, {"iters": k, "last_diff": float(last_diff)}
    return G


def polynomial_semicircle_density(
    x: float,
    a0: np.ndarray,
    A,
    *,
    eps: float = 1e-2,
    eps_reg: float | None = None,
    block_size: int = 1,
    G0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    return_info: bool = False,
) -> float:
    r'''
    Stieltjes inversion for a self-adjoint polynomial $p$ via a self-adjoint linearization.

    We evaluate at $z=x+i\,\varepsilon$ and compute the regularized fixed point
    $G(z,b_\varepsilon(z))$ associated to a self-adjoint linearization
    $L_p=a_0+\sum_i A_i\otimes X_i$.

    The scalar Cauchy transform is extracted from the distinguished corner:
    $$
      m_p(z) \approx \frac{1}{k}\,\mathrm{tr}\,\big(G(z,b_\varepsilon(z))\big)_{11},
    $$
    and the density is approximated by
    $$
      f(x) \approx -\frac{1}{\pi}\,\Im\, m_p(x+i\varepsilon).
    $$

    Parameters
    ----------
    x : float
        Real evaluation point.
    a0 : (n,n) array
        Constant term of the self-adjoint linearization.
    A : sequence of (n,n) arrays or stacked array (s,n,n)
        Coefficients defining $\eta(B)=\sum_i A_i B A_i^\ast$.
    eps : float, default 1e-2
        Imaginary offset in $z=x+i\,\varepsilon$ for Stieltjes inversion.
    eps_reg : float, optional
        Regularization used in $\Lambda_\varepsilon(z)$. If None, uses eps.
    block_size : int, default 1
        Size $k$ of the distinguished top-left block.
    G0 : (n,n) array, optional
        Warm start for the solver.
    tol : float, default 1e-10
        Relative tolerance for the fixed point.
    maxiter : int, default 10000
        Maximum iterations.
    return_info : bool, default False
        If True, also return solver diagnostics.

    Returns
    -------
    float
        Approximation to the density $f(x)$.
    '''
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    z = float(x) + 1j * float(eps)
    if eps_reg is None:
        eps_reg = float(eps)

    G, info = solve_cauchy_linearized(
        z,
        a0,
        A,
        eps_reg=eps_reg,
        block_size=block_size,
        G0=G0,
        tol=tol,
        maxiter=maxiter,
        return_info=True,
    )

    k = int(block_size)
    G11 = G[:k, :k]
    m = np.trace(G11) / k
    f = (-1.0 / np.pi) * np.imag(m)

    if return_info:
        info = dict(info)
        info["z"] = z
        info["m"] = complex(m)
        info["density"] = float(f)
        return float(f), info

    return float(f)


# Aliases
polynomial_density = polynomial_semicircle_density
get_density_C = polynomial_semicircle_density


def matrix_cauchy_semicircle_reference(w, b, *, c: float = 1.0, n_quad: int = 256):
    r"""
    Reference (brute-force) matrix Cauchy transform for a scalar semicircle.

    We compute the matrix-valued Cauchy transform
    $$
      G(w;b)\;=\;\mathbb{E}\big[(w - bS)^{-1}\big]
      \;=\;\int_{-2\sqrt{c}}^{2\sqrt{c}} (w - tb)^{-1}\, f_c(t)\,dt,
    $$
    where $S$ is a *scalar* Wigner semicircle with variance $c>0$ and density
    $$
      f_c(t)=\frac{1}{2\pi c}\sqrt{4c-t^2}\,\mathbf 1_{\{|t|\le 2\sqrt c\}}.
    $$

    For numerical stability, one typically takes $w$ in the operator upper half-plane
    (e.g. $w=zI$ with $\Im z>0$), so that all resolvents $(w-tb)^{-1}$ exist.

    Implementation detail (why this quadrature):
    Substitute $t=2\sqrt c\,x$ to get
    $$
      G(w;b)=\frac{2}{\pi}\int_{-1}^1 (w-2\sqrt c\,x\,b)^{-1}\sqrt{1-x^2}\,dx.
    $$
    The weight $\sqrt{1-x^2}$ suggests Gauss–Chebyshev quadrature of the 2nd kind:
    $$
      x_k=\cos\frac{k\pi}{n+1},\qquad
      \alpha_k=\frac{2}{n+1}\sin^2\frac{k\pi}{n+1},
    $$
    giving
    $$
      G(w;b)\approx \sum_{k=1}^n \alpha_k\,(w - (2\sqrt c\,x_k)b)^{-1}.
    $$
    The weights satisfy $\sum_k \alpha_k = 1$, so for $b=0$ the method returns $w^{-1}$
    up to floating-point rounding.

    Parameters
    ----------
    w : complex scalar or (n,n) ndarray
        Spectral parameter/matrix.
    b : (n,n) ndarray
        Deterministic matrix coefficient.
    c : float, default 1.0
        Variance of the semicircle (support is $[-2\sqrt c,2\sqrt c]$).
    n_quad : int, default 256
        Number of Chebyshev nodes (larger = more accurate, slower).

    Returns
    -------
    (n,n) ndarray
        Approximation to $G(w;b)$.
    """
    if c <= 0:
        raise ValueError("c must be > 0")
    b = np.asarray(b, dtype=np.complex128)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("b must be a square (n,n) matrix")
    n = b.shape[0]

    w_arr = np.asarray(w, dtype=np.complex128)
    w_mat = (w_arr * np.eye(n, dtype=np.complex128)) if w_arr.ndim == 0 else w_arr
    if w_mat.shape != (n, n):
        raise ValueError(f"w must be scalar or have shape {(n,n)}, got {w_mat.shape}")

    k = np.arange(1, n_quad + 1, dtype=np.float64)
    theta = k * np.pi / (n_quad + 1.0)
    x = np.cos(theta)                          # nodes in [-1,1]
    alpha = (2.0 / (n_quad + 1.0)) * (np.sin(theta) ** 2)  # weights sum to 1
    t = 2.0 * np.sqrt(c) * x

    I = np.eye(n, dtype=np.complex128)
    G = np.zeros((n, n), dtype=np.complex128)
    for ak, tk in zip(alpha, t):
        G += ak * la.solve(w_mat - tk * b, I)
    return G
