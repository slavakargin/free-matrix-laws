r"""
Brown measure of (biased) matrix-valued semicircular elements.

For arbitrary $a_0,a_1,\dots,a_r\in M_m(\mathbb C)$ and free standard
semicirculars $s_1,\dots,s_r$ let
$$
    S \;=\; a_0\otimes 1 \;+\; \sum_{i=1}^r a_i\otimes s_i .
$$
The Brown measure $\mu_S$ is recovered from the logarithmic potential
$\log\Delta(S-z)$ as $\mu_S=\frac1{2\pi}\nabla_z^2\log\Delta(S-z)$.  This
module computes the $\varepsilon$-regularized versions
$$
    L_\varepsilon(z)=\log\Delta(H_z-i\varepsilon),
    \qquad
    \rho_\varepsilon(z)=\frac1{2\pi}\nabla_z^2 L_\varepsilon(z),
$$
where $H_z=\begin{pmatrix}0&S-z\\ (S-z)^*&0\end{pmatrix}$ is the
hermitization.  As $\varepsilon\downarrow0$, $L_\varepsilon\downarrow
\log\Delta(S-z)$ and $\rho_\varepsilon\,dA\to\mu_S$ weakly.

Method
------
With $G=G(z;i\varepsilon)\in M_{2m}(\mathbb C)$ the solution of the
hermitized Dyson equation
$$
    G^{-1}+\eta^{\mathrm h}(G)=i\varepsilon-A_0(z),
    \qquad
    A_0(z)=\begin{pmatrix}0&a_0-z\\ a_0^*-\bar z&0\end{pmatrix},
    \quad
    \eta^{\mathrm h}(B)=\sum_i A_iBA_i,
    \quad
    A_i=\begin{pmatrix}0&a_i\\ a_i^*&0\end{pmatrix},
$$
one has the exact identities
$$
    \partial_z L_\varepsilon=\tfrac12\operatorname{tr}_m G_{21},
    \qquad
    \rho_\varepsilon=\frac1\pi\operatorname{tr}_m
        \bigl[\partial_{\bar z}G\bigr]_{21},
    \qquad
    (\mathrm{id}-\mathcal C_G\eta^{\mathrm h})\bigl[\partial_{\bar z}G\bigr]
      = -G\,E_{21}\,G ,
$$
where $\mathcal C_G(K)=GKG$ and $E_{21}$ is the block matrix with $I_m$
in the lower-left corner.  The operator $\mathrm{id}-\mathcal C_G\eta^{\mathrm h}$
is the Newton Jacobian of the Dyson equation (up to the factor
$\mathcal C_{G^{-1}}$), so the density costs one extra linear solve with
a factorization the Newton solver already holds.

The Dyson equation is solved by Newton's method.  Inside the support the
Jacobian has a singular value of order $\varepsilon$ (the "chiral" soft
mode $JG+GJ$, $J=\operatorname{diag}(I,-I)$); an error $\delta$ in $G$
can be amplified to $\delta/\varepsilon$ in $\partial_{\bar z}G$, which is
why a fixed-point iteration is not good enough here and the residual is
driven to machine precision.

Public API
----------
* :func:`hermitize` — build $A_0(z)$ and $(A_i)$ of size $2m$.
* :func:`cauchy_hermitized_matrix_semicircle` — Newton solver for
  $G(z;i\varepsilon)$.
* :func:`brown_log_potential` — $L_\varepsilon(z)$.
* :func:`brown_density` — $\rho_\varepsilon(z)$.
* :func:`brown_density_grid` — $\rho_\varepsilon$ on a rectangular grid
  with warm starts.
"""
from __future__ import annotations

import warnings
import numpy as np
import numpy.linalg as la
from typing import Optional

from .opvalued import _as_stack

__all__ = [
    "hermitize",
    "cauchy_hermitized_matrix_semicircle",
    "brown_log_potential",
    "brown_density",
    "brown_density_grid",
]


# ═══════════════════════════════════════════════════════════════════════════
# Hermitization and the matrix representation of eta^h
# ═══════════════════════════════════════════════════════════════════════════

def hermitize(z: complex, a0, a):
    r'''
    Hermitize the data of $S=a_0\otimes1+\sum_ia_i\otimes s_i$ at the point $z$.

    Returns the Hermitian matrices
    $$
      A_0(z)=\begin{pmatrix}0&a_0-zI\\ a_0^*-\bar zI&0\end{pmatrix},
      \qquad
      A_i=\begin{pmatrix}0&a_i\\ a_i^*&0\end{pmatrix},
      \quad i=1,\dots,r,
    $$
    of size $2m$, so that
    $H_z=\begin{pmatrix}0&S-z\\(S-z)^*&0\end{pmatrix}
     =A_0(z)\otimes1+\sum_iA_i\otimes s_i$
    is a self-adjoint matrix semicircle with mean $A_0(z)$ and covariance
    $\eta^{\mathrm h}(B)=\sum_iA_iBA_i$.

    Parameters
    ----------
    z : complex
        Point in the complex plane.
    a0 : (m,m) array_like
        Mean matrix (arbitrary, not necessarily Hermitian).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_i$ (arbitrary).

    Returns
    -------
    A0 : (2m,2m) ndarray, complex
    A : (r,2m,2m) ndarray, complex
    '''
    a0 = np.asarray(a0, dtype=complex)
    if a0.ndim != 2 or a0.shape[0] != a0.shape[1]:
        raise ValueError(f"a0 must be square; got {a0.shape!r}")
    m = a0.shape[0]
    a_arr, _ = _as_stack(a)
    a_arr = np.asarray(a_arr, dtype=complex)
    if a_arr.shape[1:] != (m, m):
        raise ValueError(f"a must be (r,{m},{m}); got {a_arr.shape!r}")

    Z = np.zeros((m, m), dtype=complex)
    I = np.eye(m, dtype=complex)
    top = a0 - complex(z) * I
    A0 = np.block([[Z, top], [top.conj().T, Z]])
    A = np.array([np.block([[Z, ai], [ai.conj().T, Z]]) for ai in a_arr])
    return A0, A


def _eta_matrix(A: np.ndarray) -> np.ndarray:
    r'''
    Matrix of $B\mapsto\sum_iA_iBA_i^*$ on $M_n(\mathbb C)$ in the
    **row-major** vectorization ``B.ravel()``:
    $\operatorname{vec}(A\,B\,C)=(A\otimes C^{\mathsf T})\operatorname{vec}(B)$.
    '''
    return sum(np.kron(Ai, Ai.conj()) for Ai in A)


def _eta_apply(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """$\\sum_i A_i B A_i^*$ for a stacked $(r,n,n)$ array $A$."""
    return np.einsum("ijk,kl,iml->jm", A, B, A.conj())


# ═══════════════════════════════════════════════════════════════════════════
# Newton solver for the hermitized Dyson equation
# ═══════════════════════════════════════════════════════════════════════════

def _is_neg_imag(G: np.ndarray) -> bool:
    """True if $-\\operatorname{Im}G\\succ0$."""
    Y = (G - G.conj().T) / (2j)
    return bool(la.eigvalsh(Y).max() < 0)


def _hfs_warmup(W: np.ndarray, A: np.ndarray, n: int, eps: float,
                G0: Optional[np.ndarray], iters: int) -> np.ndarray:
    r'''
    Damped Helton--Rashidi Far--Speicher iteration
    $G\mapsto\tfrac12\bigl[G+(W-\eta(G))^{-1}\bigr]$, used only to reach the
    basin of the physical solution before Newton.
    '''
    I = np.eye(n, dtype=complex)
    G = (-1j / max(eps, 1e-2)) * I if G0 is None else np.array(G0, dtype=complex)
    for _ in range(iters):
        G = 0.5 * (G + la.inv(W - _eta_apply(G, A)))
    return G


def _dyson_newton(W: np.ndarray, A: np.ndarray, eta_mat: np.ndarray,
                  G: np.ndarray, tol: float, maxiter: int):
    r'''
    Newton iteration for the hermitized Dyson equation in its polynomial
    form
    $$
       \tilde F(G):=I+\eta(G)\,G-W\,G=0 ,
    $$
    which is equivalent to $G^{-1}+\eta(G)=W$ for invertible $G$ but has
    no inverse, hence a bounded Hessian and a much larger Newton basin.
    Returns ``(G, iters, residual, J)`` where ``residual`` is
    $\|\tilde F(G)\|_F$ and ``J`` is the matrix of
    $K\mapsto -G^{-1}KG^{-1}+\eta(K)$ (row-major vectorization) at the final
    iterate, as needed for the density.

    **Step realization.**  At $w=0$ the equation is invariant under
    $G\mapsto VGV$, $V=\operatorname{diag}(D,D^{-1})$, for every $D\succ0$
    commuting with all $a_i$; for $D=tI$ this is exact for every $S$, and
    for nearly block-diagonal data there are further approximate symmetries
    of the same form.  Inside the support and for small $\varepsilon$ the
    Jacobian is nearly singular along the corresponding directions
    $\Xi G+G\Xi$ and the Newton direction $H$ is dominated by them; a
    straight step $G+\alpha H$ then leaves the curved orbit at second order
    and the line search stalls.  We therefore first try the plain step
    $G+\alpha H$ with at most two halvings, and if that fails we realize the
    same direction as a congruence,
    $$
       \Xi G+G\Xi=H\ \ \text{(Sylvester)},\qquad
       G_{\mathrm{new}}=e^{\alpha\Xi}\,G\,e^{\alpha\Xi},
    $$
    which follows the scaling orbits exactly.  Positivity of
    $-\operatorname{Im}G$ is not enforced along the way (outside the
    spectrum it is only $O(\varepsilon)$ and a guard blocks every useful
    step); the caller verifies it at the end and falls back to the damped
    fixed-point iteration, which preserves the half-plane, if Newton has
    landed on an unphysical branch.
    '''
    from scipy.linalg import solve_sylvester, expm
    n = G.shape[0]
    I = np.eye(n, dtype=complex)

    def resid(G):
        return I + _eta_apply(G, A) @ G - W @ G

    F = resid(G)
    res = la.norm(F)
    for it in range(maxiter):
        if res <= tol:
            break
        M = _eta_apply(G, A) - W
        J = np.kron(I, G.T) @ eta_mat + np.kron(M, I)
        try:
            H = la.solve(J, -F.ravel()).reshape(n, n)
        except la.LinAlgError:
            break
        ok = False
        alpha = 1.0
        for _ in range(3):                      # plain step
            G_try = G + alpha * H
            F_try = resid(G_try)
            res_try = la.norm(F_try)
            if res_try < res:
                ok = True
                break
            alpha *= 0.5
        if not ok:                               # congruence step
            Xi = solve_sylvester(G, G, H)
            if np.isfinite(Xi).all() and la.norm(Xi) < 30.0:
                alpha = 1.0
                for _ in range(40):
                    E = expm(alpha * Xi)
                    G_try = E @ G @ E
                    F_try = resid(G_try)
                    res_try = la.norm(F_try)
                    if res_try < res:
                        ok = True
                        break
                    alpha *= 0.5
        if not ok:                               # last resort: tiny plain steps
            alpha = 0.125
            for _ in range(30):
                G_try = G + alpha * H
                F_try = resid(G_try)
                res_try = la.norm(F_try)
                if res_try < res:
                    ok = True
                    break
                alpha *= 0.5
        if not ok:
            break
        G, F, res = G_try, F_try, res_try
    else:
        it = maxiter

    try:
        Ginv = la.inv(G)
        J = -np.kron(Ginv, Ginv.T) + eta_mat
    except la.LinAlgError:
        J = None
    return G, it, res, J


def cauchy_hermitized_matrix_semicircle(
    z: complex,
    a0,
    a,
    eps: float = 1e-4,
    G0: Optional[np.ndarray] = None,
    tol: float = 1e-13,
    maxiter: int = 100,
    warmup: int = 200,
    return_info: bool = False,
):
    r'''
    Hermitized matrix Cauchy transform $G(z;i\varepsilon)$ of
    $S=a_0\otimes1+\sum_ia_i\otimes s_i$.

    Solves, by Newton's method,
    $$
      G^{-1}+\eta^{\mathrm h}(G)=i\varepsilon I_{2m}-A_0(z),
      \qquad -\operatorname{Im}G\succ0 ,
    $$
    where $A_0(z)$ and $\eta^{\mathrm h}$ are built by :func:`hermitize`.
    In block form,
    $$
      G(z;i\varepsilon)=
      \begin{pmatrix}
        -i\varepsilon\,\mathbb E[(XX^*+\varepsilon^2)^{-1}] &
        -\mathbb E[X(X^*X+\varepsilon^2)^{-1}]\\
        -\mathbb E[X^*(XX^*+\varepsilon^2)^{-1}] &
        -i\varepsilon\,\mathbb E[(X^*X+\varepsilon^2)^{-1}]
      \end{pmatrix},
      \qquad X=S-z,
    $$
    with $\mathbb E=\mathrm{id}\otimes\tau$.  In particular
    $G_{12}=G_{21}^*$ and $\operatorname{tr}G_{11}=\operatorname{tr}G_{22}$.

    Parameters
    ----------
    z : complex
    a0 : (m,m) array_like
    a : sequence of (m,m) arrays or stacked (r,m,m) array
    eps : float, default 1e-4
        Regularization $\varepsilon>0$.
    G0 : (2m,2m) ndarray, optional
        Warm start (e.g. the solution at a nearby $z$).  If given, the
        damped warm-up iteration is skipped.
    tol : float, default 1e-13
        Frobenius norm of the Dyson residual at which Newton stops.
    maxiter : int, default 100
        Newton iteration cap.
    warmup : int, default 200
        Number of damped fixed-point steps before Newton when ``G0`` is
        ``None``.
    return_info : bool, default False
        If True, also return a dict with keys ``iters``, ``residual`` and
        ``jacobian`` (the $(2m)^2\times(2m)^2$ matrix
        $-G^{-1}\otimes G^{-\mathsf T}+\eta^{\mathrm h}$ in row-major
        vectorization).

    Returns
    -------
    G : (2m,2m) ndarray, complex
    info : dict (only if ``return_info``)
    '''
    if eps <= 0:
        raise ValueError("eps must be > 0")
    A0, A = hermitize(z, a0, a)
    n = A0.shape[0]
    W = 1j * eps * np.eye(n, dtype=complex) - A0
    eta_mat = _eta_matrix(A)

    if G0 is None:
        G = _hfs_warmup(W, A, n, eps, None, warmup)
    else:
        G = np.array(G0, dtype=complex)
        if not _is_neg_imag(G):
            G = _hfs_warmup(W, A, n, eps, None, warmup)

    G, iters, res, J = _dyson_newton(W, A, eta_mat, G, tol, maxiter)
    if res > 1e3 * tol or not _is_neg_imag(G):
        # fall back to a long damped iteration (which preserves the
        # half-plane), then polish
        G = _hfs_warmup(W, A, n, eps, None, 20 * warmup)
        G, iters2, res, J = _dyson_newton(W, A, eta_mat, G, tol, maxiter)
        iters += iters2
    if res > 1e-8:
        warnings.warn(f"Dyson solver stopped with residual {res:.2e} at z={z}, "
                      f"eps={eps}; the result may be inaccurate.", RuntimeWarning)
    if not _is_neg_imag(G):
        raise RuntimeError(
            "Dyson solver did not return a solution with -Im G > 0 "
            f"(residual {res:.2e}); try a larger eps or a warm start.")

    if return_info:
        return G, {"iters": iters, "residual": float(res), "jacobian": J}
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Logarithmic potential and density
# ═══════════════════════════════════════════════════════════════════════════

def _log_potential_from_G(G: np.ndarray, A: np.ndarray) -> float:
    n = G.shape[0]
    _, logabsdet = la.slogdet(G)
    trace_term = np.trace(G @ _eta_apply(G, A)) / n
    return float(-logabsdet / n + 0.5 * np.real(trace_term))


def brown_log_potential(
    z: complex,
    a0,
    a,
    eps: float = 1e-4,
    G0: Optional[np.ndarray] = None,
    tol: float = 1e-13,
    maxiter: int = 100,
) -> float:
    r'''
    Regularized logarithmic potential of the Brown measure,
    $$
      L_\varepsilon(z)=\log\Delta(H_z-i\varepsilon)
      =\tfrac12(\operatorname{tr}_m\otimes\tau)\log\bigl((S-z)^*(S-z)+\varepsilon^2\bigr),
    $$
    which decreases to $\log\Delta(S-z)=\int\log|w-z|\,d\mu_S(w)$ as
    $\varepsilon\downarrow0$.

    Computed from $G=G(z;i\varepsilon)$ by the closed determinant identity
    $$
      L_\varepsilon(z)=-\frac1{2m}\log|\det G|
        +\frac12\operatorname{Re}\operatorname{tr}_{2m}\bigl(G\,\eta^{\mathrm h}(G)\bigr).
    $$

    Parameters
    ----------
    z, a0, a, eps, G0, tol, maxiter
        As in :func:`cauchy_hermitized_matrix_semicircle`.

    Returns
    -------
    float
    '''
    _, A = hermitize(z, a0, a)
    G = cauchy_hermitized_matrix_semicircle(z, a0, a, eps=eps, G0=G0,
                                             tol=tol, maxiter=maxiter)
    return _log_potential_from_G(G, A)


def _density_from_G(G: np.ndarray, J: np.ndarray, eta_mat: np.ndarray,
                    m: int):
    r'''
    Density from the solved $G$ and the Newton Jacobian
    $J=-G^{-1}\otimes G^{-\mathsf T}+\eta$.

    $\partial_{\bar z}G$ solves
    $-G^{-1}KG^{-1}+\eta(K)=E_{21}$, i.e. $J\,\operatorname{vec}K=\operatorname{vec}E_{21}$,
    and $\rho=\frac1\pi\operatorname{tr}_mK_{21}$.
    '''
    n = 2 * m
    E21 = np.zeros((n, n), dtype=complex)
    E21[m:, :m] = np.eye(m)
    K = la.solve(J, E21.ravel()).reshape(n, n)
    val = np.trace(K[m:, :m]) / m / np.pi
    return val, K


def brown_density(
    z: complex,
    a0,
    a,
    eps: float = 1e-4,
    G0: Optional[np.ndarray] = None,
    tol: float = 1e-13,
    maxiter: int = 100,
    return_info: bool = False,
):
    r'''
    Regularized density $\rho_\varepsilon(z)$ of the Brown measure of
    $S=a_0\otimes1+\sum_ia_i\otimes s_i$.

    $$
      \rho_\varepsilon(z)=\frac1{2\pi}\nabla_z^2 L_\varepsilon(z)
      =\frac1\pi\operatorname{tr}_m\bigl[\partial_{\bar z}G\bigr]_{21},
      \qquad
      (\mathrm{id}-\mathcal C_G\eta^{\mathrm h})\bigl[\partial_{\bar z}G\bigr]
        =-G\,E_{21}\,G,
    $$
    with $G=G(z;i\varepsilon)$ from
    :func:`cauchy_hermitized_matrix_semicircle`.  The identity is exact for
    every $\varepsilon>0$; $\rho_\varepsilon\,dA$ converges weakly to the
    Brown measure as $\varepsilon\downarrow0$.  The density of the Brown
    measure typically jumps at the boundary of its support, and the
    regularization smooths that jump over a scale of order
    $\varepsilon$; in the bulk the bias is $O(\varepsilon^2)$.

    Parameters
    ----------
    z : complex
    a0 : (m,m) array_like
        Mean matrix (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients (arbitrary; no self-adjointness assumed).
    eps : float, default 1e-4
    G0 : (2m,2m) ndarray, optional
        Warm start for the Dyson solver.
    tol, maxiter
        Newton parameters; see :func:`cauchy_hermitized_matrix_semicircle`.
    return_info : bool, default False
        If True, also return a dict with ``G`` (for warm-starting a
        neighbouring point), ``dG_dzbar``, ``log_potential``,
        ``residual`` and ``iters``.

    Returns
    -------
    float
        $\rho_\varepsilon(z)$.  The computed value is real up to round-off;
        its imaginary part is discarded.
    info : dict (only if ``return_info``)

    Examples
    --------
    Circular element ($a_1=1/\sqrt2$, $a_2=i/\sqrt2$): uniform on the unit disk.

    >>> import numpy as np
    >>> from free_matrix_laws import brown_density
    >>> a0 = np.zeros((1, 1)); a = [np.eye(1)/np.sqrt(2), 1j*np.eye(1)/np.sqrt(2)]
    >>> round(brown_density(0.3+0.2j, a0, a, eps=1e-4) * np.pi, 3)
    1.0
    '''
    a0 = np.asarray(a0, dtype=complex)
    m = a0.shape[0]
    G, info = cauchy_hermitized_matrix_semicircle(
        z, a0, a, eps=eps, G0=G0, tol=tol, maxiter=maxiter, return_info=True)
    _, A = hermitize(z, a0, a)
    val, K = _density_from_G(G, info["jacobian"], None, m)
    rho = float(np.real(val))
    if return_info:
        out = {
            "G": G,
            "dG_dzbar": K,
            "log_potential": _log_potential_from_G(G, A),
            "residual": info["residual"],
            "iters": info["iters"],
            "imag_part": float(np.imag(val)),
        }
        return rho, out
    return rho


def brown_density_grid(
    xs,
    ys,
    a0,
    a,
    eps: float = 1e-4,
    tol: float = 1e-13,
    maxiter: int = 100,
    return_potential: bool = False,
):
    r'''
    $\rho_\varepsilon$ on the rectangular grid $\{x_j+iy_k\}$.

    The grid is traversed row by row in serpentine order and each point is
    warm-started from its predecessor.

    Parameters
    ----------
    xs, ys : 1-D array_like
        Real and imaginary coordinates.
    a0, a, eps, tol, maxiter
        As in :func:`brown_density`.
    return_potential : bool, default False
        If True, also return $L_\varepsilon$ on the grid.

    Returns
    -------
    rho : (len(ys), len(xs)) ndarray
        ``rho[k, j]`` is $\rho_\varepsilon(x_j+iy_k)$.
    L : (len(ys), len(xs)) ndarray (only if ``return_potential``)
    '''
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rho = np.empty((ys.size, xs.size))
    L = np.empty_like(rho)
    G_prev = None
    for k, y in enumerate(ys):
        cols = range(xs.size) if k % 2 == 0 else range(xs.size - 1, -1, -1)
        for j in cols:
            val, info = brown_density(xs[j] + 1j * y, a0, a, eps=eps,
                                      G0=G_prev, tol=tol, maxiter=maxiter,
                                      return_info=True)
            rho[k, j] = val
            L[k, j] = info["log_potential"]
            G_prev = info["G"]
    if return_potential:
        return rho, L
    return rho
