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
* [`hermitize`][free_matrix_laws.brown.hermitize] — build $A_0(z)$ and $(A_i)$ of size $2m$.
* [`cauchy_hermitized_matrix_semicircle`][free_matrix_laws.brown.cauchy_hermitized_matrix_semicircle] — Newton solver for
  $G(z;i\varepsilon)$.
* [`brown_log_potential`][free_matrix_laws.brown.brown_log_potential] — $L_\varepsilon(z)$.
* [`brown_density`][free_matrix_laws.brown.brown_density] — $\rho_\varepsilon(z)$.
* [`brown_density_grid`][free_matrix_laws.brown.brown_density_grid] — $\rho_\varepsilon$ on a rectangular grid
  with warm starts.

Spectrum criterion
------------------
This is a criterion in size $m$ for the spectrum of $S$, with no regularization.
For $z\notin\sigma(S)$ the resolvent $g(z)=(\mathrm{id}\otimes\tau)[(z-S)^{-1}]$
solves $g^{-1}+\kappa(g)=z-a_0$ with $\kappa(B)=\sum_ia_iBa_i$, and the
map $T_g(B)=\sum_i(ga_i)B(ga_i)^*$ has spectral radius $r(T_g)<1$;
conversely any invertible solution with $r(T_g)<1$ certifies
$z\notin\sigma(S)$.  The boundary of the spectrum lies on the level set
$r(T_{g(z)})=1$ (or where $g$ blows up).  The support of the Brown
measure is contained in $\sigma(S)$; whether they coincide is open.

* [`resolvent_matrix_semicircle`][free_matrix_laws.brown.resolvent_matrix_semicircle]
  — $g(z)$ by Newton in size $m$.
* [`spectral_radius_T`][free_matrix_laws.brown.spectral_radius_T] — $r(T_g)$.
* [`spectrum_indicator`][free_matrix_laws.brown.spectrum_indicator] —
  $r(T_{g(z)})$, with $+\infty$ where no solution exists; $<1$ exactly
  off the spectrum.
* [`spectrum_indicator_grid`][free_matrix_laws.brown.spectrum_indicator_grid],
  [`spectrum_boundary`][free_matrix_laws.brown.spectrum_boundary] —
  the indicator on a grid and the boundary by bisection on grid edges.
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
    "resolvent_matrix_semicircle",
    "spectral_radius_T",
    "spectrum_indicator",
    "spectrum_indicator_grid",
    "spectrum_boundary",
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

    $$
      H_z=\begin{pmatrix}0&S-z\\(S-z)^*&0\end{pmatrix}
         =A_0(z)\otimes1+\sum_iA_i\otimes s_i
    $$

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
    A0 : (2m,2m) ndarray
        The complex Hermitian matrix $A_0(z)$.
    A : (r,2m,2m) ndarray
        The complex Hermitian matrices $A_1,\dots,A_r$, stacked.
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

    where $A_0(z)$ and $\eta^{\mathrm h}$ are built by [`hermitize`][free_matrix_laws.brown.hermitize].
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
        Point $z\in\mathbb C$ at which $S-z$ is hermitized.
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary, not necessarily Hermitian).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary).
    eps : float, default 1e-4
        Regularization $\varepsilon>0$.
    G0 : (2m,2m) ndarray, optional
        Warm start, e.g. the solution at a nearby $z$.  If it satisfies
        $-\operatorname{Im}G_0\succ0$, the damped warm-up iteration is
        skipped.
    tol : float, default 1e-13
        Newton stops when the Frobenius norm of the residual
        $I+\eta^{\mathrm h}(G)\,G-W\,G$, $W=i\varepsilon-A_0(z)$, is
        below `tol`.
    maxiter : int, default 100
        Newton iteration cap.
    warmup : int, default 200
        Number of damped fixed-point steps before Newton when `G0` is
        `None` (twenty times as many in the fallback).
    return_info : bool, default False
        If True, also return a dict with keys `iters`, `residual` and
        `jacobian` (the $(2m)^2\times(2m)^2$ matrix of
        $K\mapsto-G^{-1}KG^{-1}+\eta^{\mathrm h}(K)$ in row-major
        vectorization).

    Returns
    -------
    G : (2m,2m) ndarray
        The complex matrix $G(z;i\varepsilon)$.
    info : dict
        Returned only if `return_info` is True; see above.

    Raises
    ------
    RuntimeError
        If neither Newton nor the damped fallback produce a solution with
        $-\operatorname{Im}G\succ0$.

    Warns
    -----
    RuntimeWarning
        If the final residual exceeds $10^{-8}$.
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
    z : complex
        Point in the complex plane.
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary).
    eps : float, default 1e-4
        Regularization $\varepsilon>0$.
    G0 : (2m,2m) ndarray, optional
        Warm start for the Dyson solver.
    tol : float, default 1e-13
        Residual tolerance; see [`cauchy_hermitized_matrix_semicircle`][free_matrix_laws.brown.cauchy_hermitized_matrix_semicircle].
    maxiter : int, default 100
        Newton iteration cap.

    Returns
    -------
    float
        $L_\varepsilon(z)$.
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
    [`cauchy_hermitized_matrix_semicircle`][free_matrix_laws.brown.cauchy_hermitized_matrix_semicircle].  The identity is exact for
    every $\varepsilon>0$; $\rho_\varepsilon\,dA$ converges weakly to the
    Brown measure as $\varepsilon\downarrow0$.  The density of the Brown
    measure typically jumps at the boundary of its support, and the
    regularization smooths that jump over a scale of order
    $\varepsilon$; in the bulk the bias is $O(\varepsilon^2)$.

    Parameters
    ----------
    z : complex
        Point in the complex plane.
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary; no self-adjointness
        assumed).
    eps : float, default 1e-4
        Regularization $\varepsilon>0$.
    G0 : (2m,2m) ndarray, optional
        Warm start for the Dyson solver.
    tol : float, default 1e-13
        Residual tolerance; see [`cauchy_hermitized_matrix_semicircle`][free_matrix_laws.brown.cauchy_hermitized_matrix_semicircle].
    maxiter : int, default 100
        Newton iteration cap.
    return_info : bool, default False
        If True, also return a dict with keys `G` (for warm-starting a
        neighbouring point), `dG_dzbar`, `log_potential`, `residual`,
        `iters` and `imag_part`.

    Returns
    -------
    rho : float
        $\rho_\varepsilon(z)$.  The computed value is real up to
        round-off; its imaginary part is discarded (and reported as
        `imag_part` in `info`).
    info : dict
        Returned only if `return_info` is True; see above.

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
    xs : 1-D array_like
        Real parts $x_j$.
    ys : 1-D array_like
        Imaginary parts $y_k$.
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary).
    eps : float, default 1e-4
        Regularization $\varepsilon>0$.
    tol : float, default 1e-13
        Residual tolerance; see [`cauchy_hermitized_matrix_semicircle`][free_matrix_laws.brown.cauchy_hermitized_matrix_semicircle].
    maxiter : int, default 100
        Newton iteration cap.
    return_potential : bool, default False
        If True, also return $L_\varepsilon$ on the grid.

    Returns
    -------
    rho : (len(ys), len(xs)) ndarray
        `rho[k, j]` is $\rho_\varepsilon(x_j+iy_k)$.
    L : (len(ys), len(xs)) ndarray
        $L_\varepsilon(x_j+iy_k)$; returned only if `return_potential`
        is True.
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


# ═══════════════════════════════════════════════════════════════════════════
# Spectrum of S in size m: resolvent g(z) and the criterion r(T_g) < 1
# ═══════════════════════════════════════════════════════════════════════════

def _kappa_apply(B: np.ndarray, a: np.ndarray) -> np.ndarray:
    r"""$\kappa(B)=\sum_i a_iBa_i$ (no adjoint) for a stacked $(r,m,m)$ array."""
    return np.einsum("ijk,kl,ilm->jm", a, B, a)


def _kappa_matrix(a: np.ndarray) -> np.ndarray:
    r"""Matrix of $\kappa$ in row-major vectorization: $\sum_i a_i\otimes a_i^{\mathsf T}$."""
    return sum(np.kron(ai, ai.T) for ai in a)


def spectral_radius_T(g: np.ndarray, a) -> float:
    r'''
    Spectral radius of the completely positive map
    $T_g(B)=g\,\eta_1(B)\,g^*=\sum_i (ga_i)\,B\,(ga_i)^*$ on $M_m(\mathbb C)$.

    By the Perron--Frobenius theorem for positive maps the spectral radius
    is an eigenvalue with a positive semidefinite eigenmatrix.  It is
    computed as the largest modulus of the eigenvalues of the
    $m^2\times m^2$ matrix $\sum_i (ga_i)\otimes\overline{(ga_i)}$.

    Parameters
    ----------
    g : (m,m) ndarray
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$.

    Returns
    -------
    float
        $r(T_g)$.
    '''
    a_arr, _ = _as_stack(a)
    g = np.asarray(g, dtype=complex)
    L = [g @ np.asarray(ai, dtype=complex) for ai in a_arr]
    T = sum(np.kron(Li, Li.conj()) for Li in L)
    return float(np.abs(la.eigvals(T)).max())


def resolvent_matrix_semicircle(
    z: complex,
    a0,
    a,
    g0: Optional[np.ndarray] = None,
    eps: float = 1e-4,
    tol: float = 1e-13,
    maxiter: int = 60,
    return_info: bool = False,
):
    r'''
    Matrix resolvent $g(z)=(\mathrm{id}\otimes\tau)\bigl[(z-S)^{-1}\bigr]\in M_m(\mathbb C)$
    of $S=a_0\otimes1+\sum_ia_i\otimes s_i$ for $z$ outside the spectrum.

    Outside the spectrum $g$ is the unique invertible solution of the
    size-$m$ equation

    $$
      g^{-1}+\kappa(g)=z-a_0,
      \qquad
      \kappa(B)=\sum_i a_iBa_i \quad\text{(no adjoint)},
    $$

    with $r(T_g)<1$, where $T_g(B)=\sum_i(ga_i)B(ga_i)^*$; see
    [`spectral_radius_T`][free_matrix_laws.brown.spectral_radius_T] and
    [`spectrum_indicator`][free_matrix_laws.brown.spectrum_indicator].
    The equation is solved by Newton's method on the polynomial residual
    $I+\kappa(g)g-(z-a_0)g$.  Unless a warm start `g0` is given, the
    iteration starts from the $(2,1)$ block of the hermitized Cauchy
    transform $G(z;i\varepsilon)$, which converges to $g(z)$ as
    $\varepsilon\downarrow0$ on every component of the resolvent set
    (including bounded ones, where $g$ is not the continuation from
    infinity).

    Inside the spectrum the equation has no solution of the required
    kind; Newton may then fail or converge to a solution with
    $r(T_g)\ge1$.  Either outcome is reported through `info` and is what
    the spectrum criterion relies on.

    Parameters
    ----------
    z : complex
        Point in the complex plane.
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary).
    g0 : (m,m) ndarray, optional
        Warm start, e.g. $g$ at a nearby point outside the spectrum.
    eps : float, default 1e-4
        Regularization of the hermitized solve used for the cold start
        (only a starting point: the result does not depend on it).
    tol : float, default 1e-13
        Newton stops when the Frobenius norm of the residual is below
        `tol`.
    maxiter : int, default 60
        Newton iteration cap.
    return_info : bool, default False
        If True, also return a dict with keys `residual`, `iters` and
        `converged`.

    Returns
    -------
    g : (m,m) ndarray
        The last Newton iterate (equal to $g(z)$ when `converged`).
    info : dict
        Returned only if `return_info` is True.
    '''
    a0 = np.asarray(a0, dtype=complex)
    m = a0.shape[0]
    a_arr, _ = _as_stack(a)
    a_arr = np.asarray(a_arr, dtype=complex)
    I = np.eye(m, dtype=complex)
    W = complex(z) * I - a0
    kmat = _kappa_matrix(a_arr)

    if g0 is None:
        try:
            G = cauchy_hermitized_matrix_semicircle(z, a0, a, eps=eps)
            g = np.array(G[m:, :m])
        except RuntimeError:
            # no physical hermitized solution found: z is (numerically)
            # deep inside the spectrum; start from the free-resolvent guess
            g = la.inv(W)
    else:
        g = np.array(g0, dtype=complex)

    def resid(g):
        return I + _kappa_apply(g, a_arr) @ g - W @ g

    F = resid(g)
    res = la.norm(F)
    converged = res <= tol
    it = 0
    while not converged and it < maxiter:
        M = _kappa_apply(g, a_arr) - W
        J = np.kron(I, g.T) @ kmat + np.kron(M, I)
        try:
            H = la.solve(J, -F.ravel()).reshape(m, m)
        except la.LinAlgError:
            break
        alpha, ok = 1.0, False
        for _ in range(30):
            g_try = g + alpha * H
            F_try = resid(g_try)
            res_try = la.norm(F_try)
            if res_try < res:
                ok = True
                break
            alpha *= 0.5
        if not ok:
            break
        g, F, res = g_try, F_try, res_try
        it += 1
        converged = res <= tol

    if return_info:
        return g, {"residual": float(res), "iters": it, "converged": bool(converged)}
    return g


def spectrum_indicator(
    z: complex,
    a0,
    a,
    g0: Optional[np.ndarray] = None,
    eps: float = 1e-4,
    tol: float = 1e-11,
    return_g: bool = False,
):
    r'''
    Spectrum criterion for $S=a_0\otimes1+\sum_ia_i\otimes s_i$ at $z$.

    Returns $r(T_g)$ for the solution $g$ of the size-$m$ resolvent
    equation found by
    [`resolvent_matrix_semicircle`][free_matrix_laws.brown.resolvent_matrix_semicircle],
    or $+\infty$ if no solution was found.  Then

    $$
      z\notin\sigma(S)
      \quad\Longleftrightarrow\quad
      \texttt{spectrum\_indicator}(z)<1 .
    $$

    Indeed, $z\notin\sigma(S)$ implies that $g(z)$ solves the equation
    with $r(T_{g(z)})<1$, and conversely any invertible solution with
    $r(T_g)<1$ forces $z\notin\sigma(S)$ and equals $g(z)$; so inside
    the spectrum every solution Newton can land on has $r(T_g)\ge1$.
    The boundary of the spectrum is contained in the level set
    $\{r(T_{g(z)})=1\}$ together with points where $\|g(z)\|\to\infty$.

    The support of the Brown measure is contained in $\sigma(S)$; whether
    the two coincide for matrix semicirculars is an open question, so this
    function describes the spectrum, which may be larger.

    Parameters
    ----------
    z : complex
    a0 : (m,m) array_like
        Mean matrix $a_0$ (arbitrary).
    a : sequence of (m,m) arrays or stacked (r,m,m) array
        Coefficients $a_1,\dots,a_r$ (arbitrary).
    g0 : (m,m) ndarray, optional
        Warm start for the resolvent solve.
    eps : float, default 1e-4
        Regularization for the cold start (hermitized solve).
    tol : float, default 1e-11
        Residual tolerance below which the solve counts as converged.
    return_g : bool, default False
        If True, also return the matrix $g$ (or `None` if not converged).

    Returns
    -------
    float
        $r(T_g)$, or `inf` if the resolvent equation was not solved.
    g : (m,m) ndarray or None
        Returned only if `return_g` is True.
    '''
    g, info = resolvent_matrix_semicircle(z, a0, a, g0=g0, eps=eps, tol=tol,
                                          return_info=True)
    if info["converged"] and np.isfinite(g).all():
        r = spectral_radius_T(g, a)
    else:
        r, g = np.inf, None
    if return_g:
        return r, g
    return r


def spectrum_indicator_grid(xs, ys, a0, a, eps: float = 1e-4, tol: float = 1e-11):
    r'''
    [`spectrum_indicator`][free_matrix_laws.brown.spectrum_indicator] on the
    rectangular grid $\{x_j+iy_k\}$; each point is cold-started from the
    hermitized solve, which is itself warm-started along a serpentine path.

    Parameters
    ----------
    xs, ys : 1-D array_like
        Real and imaginary coordinates.
    a0, a, eps, tol
        As in [`spectrum_indicator`][free_matrix_laws.brown.spectrum_indicator].

    Returns
    -------
    r : (len(ys), len(xs)) ndarray
        `r[k, j]` is the indicator at $x_j+iy_k$ (`inf` where the
        resolvent equation was not solved).  `r < 1` marks the
        complement of the spectrum.
    '''
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    a0 = np.asarray(a0, dtype=complex)
    m = a0.shape[0]
    out = np.empty((ys.size, xs.size))
    G_prev = None
    for k, y in enumerate(ys):
        cols = range(xs.size) if k % 2 == 0 else range(xs.size - 1, -1, -1)
        for j in cols:
            z = xs[j] + 1j * y
            try:
                G_prev = cauchy_hermitized_matrix_semicircle(z, a0, a, eps=eps, G0=G_prev)
                g0 = G_prev[m:, :m]
            except RuntimeError:
                G_prev, g0 = None, None
            out[k, j] = spectrum_indicator(z, a0, a, g0=g0, eps=eps, tol=tol)
    return out


def spectrum_boundary(xs, ys, a0, a, eps: float = 1e-4, tol: float = 1e-11,
                      xtol: float = 1e-6, return_grid: bool = False):
    r'''
    Points on the boundary of the spectrum of $S=a_0\otimes1+\sum_ia_i\otimes s_i$.

    The indicator $r(T_{g(z)})$ is evaluated on the grid
    $\{x_j+iy_k\}$; on every grid edge whose endpoints are classified
    differently (one with $r<1$, the other with $r\ge1$ or no solution)
    the boundary is located by bisection, warm-starting the resolvent
    solve from the exterior endpoint so that the iteration stays on the
    physical branch.  The bisection stops when the edge is shorter than
    `xtol`.  Points at which $\|g\|$ blows up (Corollary of the
    criterion: the alternative to $r(T_g)=1$) are handled by the same
    rule, since a failed solve counts as "inside".

    Features thinner than the grid spacing are missed; refine the grid
    near the real axis when the coefficients are real.

    Parameters
    ----------
    xs, ys : 1-D array_like
        Grid coordinates.
    a0, a, eps, tol
        As in [`spectrum_indicator`][free_matrix_laws.brown.spectrum_indicator].
    xtol : float, default 1e-6
        Bisection tolerance in $z$.
    return_grid : bool, default False
        If True, also return the grid indicator array.

    Returns
    -------
    pts : (n,) ndarray of complex
        Boundary points (unordered).
    r : (len(ys), len(xs)) ndarray
        Returned only if `return_grid` is True.
    '''
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    r = spectrum_indicator_grid(xs, ys, a0, a, eps=eps, tol=tol)
    outside = r < 1.0
    pts = []

    def bisect(z_out, z_in):
        g_out = spectrum_indicator(z_out, a0, a, eps=eps, tol=tol, return_g=True)[1]
        while abs(z_in - z_out) > xtol:
            zm = 0.5 * (z_out + z_in)
            rm, gm = spectrum_indicator(zm, a0, a, g0=g_out, eps=eps, tol=tol, return_g=True)
            if rm < 1.0:
                z_out, g_out = zm, gm
            else:
                z_in = zm
        return 0.5 * (z_out + z_in)

    for k in range(ys.size):
        for j in range(xs.size):
            z = xs[j] + 1j * ys[k]
            if j + 1 < xs.size and outside[k, j] != outside[k, j + 1]:
                z2 = xs[j + 1] + 1j * ys[k]
                pts.append(bisect(z, z2) if outside[k, j] else bisect(z2, z))
            if k + 1 < ys.size and outside[k, j] != outside[k + 1, j]:
                z2 = xs[j] + 1j * ys[k + 1]
                pts.append(bisect(z, z2) if outside[k, j] else bisect(z2, z))
    pts = np.array(pts, dtype=complex)
    if return_grid:
        return pts, r
    return pts
