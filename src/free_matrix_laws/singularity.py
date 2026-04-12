r"""
Singularity analysis for the HMS subordination function W(u).

Given Kraus operators $A_1,\dots,A_r$, the HMS equation is
$$
    \eta(W)\,W + u\,W = I, \qquad \Re\,u > 0,
$$
where $\eta(B) = \sum_i A_i B A_i^*$.  Near a singularity at $u=0$,
the key spectral quantities—$\operatorname{tr} W(u)$, eigenvalues of $W(u)$,
$\det W(u)$—follow Puiseux asymptotics $\sim C\,u^\alpha$.

This module provides:

* :func:`solve_hms` — Newton solver for the HMS equation at scalar $u$.
* :func:`hms_sweep` — warm-started sweep over a geometric grid of $u$ values.
* :func:`puiseux_exponent` — log-log regression estimate of the leading
  Puiseux exponent from sweep data.
* :func:`singularity_report` — one-call summary combining all of the above.
"""

from __future__ import annotations

import warnings
import numpy as np
import numpy.linalg as la
from typing import Optional

from .opvalued import covariance_map as _eta


# ═══════════════════════════════════════════════════════════════════════
# Newton solver for eta(W)W + uW = I
# ═══════════════════════════════════════════════════════════════════════

def _build_eta_matrix(A, n: int) -> np.ndarray:
    """Build the n^2 x n^2 matrix representation of eta."""
    n2 = n * n
    eta_mat = np.zeros((n2, n2), dtype=complex)
    for j in range(n2):
        E = np.zeros((n, n), dtype=complex)
        E.flat[j] = 1.0
        eta_mat[:, j] = _eta(E, A).ravel()
    return eta_mat


def solve_hms(
    u: float,
    A,
    W0: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    maxiter: int = 300,
    _eta_mat: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, float]:
    r"""
    Solve $\eta(W)W + uW = I$ by Newton's method with backtracking.

    Parameters
    ----------
    u : float
        Positive real parameter.
    A : list of (n, n) arrays or stacked (s, n, n) array
        Kraus operators defining $\eta$.
    W0 : (n, n) array, optional
        Initial guess (default: $I/(u+1)$).
    tol : float
        Residual tolerance.
    maxiter : int
        Maximum Newton iterations.
    _eta_mat : (n^2, n^2) array, optional
        Precomputed matrix representation of $\eta$ (for speed in sweeps).

    Returns
    -------
    W : (n, n) ndarray
        Solution matrix.
    iters : int
        Number of iterations used.
    residual : float
        Final residual $\|F(W)\|$.
    """
    n = A[0].shape[0] if isinstance(A, (list, tuple)) else A.shape[-1]
    I_n = np.eye(n, dtype=complex)

    if W0 is None:
        W = I_n / (u + 1.0)
    else:
        W = W0.copy().astype(complex)

    if _eta_mat is None:
        _eta_mat = _build_eta_matrix(A, n)

    for it in range(maxiter):
        etaW = _eta(W, A)
        F = etaW @ W + u * W - I_n
        res = la.norm(F)
        if res < tol:
            return W, it, res

        # Jacobian: D_W F[H] = eta(H) W + (eta(W) + u I) H
        J = (np.kron(W.T, I_n) @ _eta_mat
             + np.kron(I_n, etaW + u * I_n))
        H = la.solve(J, -F.ravel()).reshape(n, n)

        # Backtracking line search
        alpha = 1.0
        for _ in range(30):
            W_try = W + alpha * H
            F_try = _eta(W_try, A) @ W_try + u * W_try - I_n
            if la.norm(F_try) < res:
                break
            alpha *= 0.5
        else:
            break  # line search failed

        W = W + alpha * H

    return W, maxiter, la.norm(_eta(W, A) @ W + u * W - I_n)


# ═══════════════════════════════════════════════════════════════════════
# Warm-started sweep
# ═══════════════════════════════════════════════════════════════════════

def hms_sweep(
    A,
    u_max: float | None = None,
    u_min: float = 1e-6,
    n_points: int = 300,
    tol: float = 1e-13,
    maxiter: int = 20000,
    convergence_threshold: float = 1e-8,
    method: str = "hfs",
    progress: bool | int = False,
) -> dict:
    r"""
    Sweep the HMS equation from large to small $u$ with warm-starting.

    Parameters
    ----------
    A : list of (n, n) arrays or (s, n, n) array
        Kraus operators.
    u_max : float or None
        Upper bound for $u$. If None, automatically set to
        $3\,\|\eta(I)\|_{\mathrm{op}}$ so the solver starts in the
        easy asymptotic regime $W \approx I/u$.
    u_min : float
        Lower bound for $u$.
    n_points : int
        Number of grid points.
    tol : float
        Solver tolerance.
    maxiter : int
        Max iterations per $u$.
    convergence_threshold : float
        Points with residual above this are flagged as not converged.
    method : ``"hfs"`` or ``"newton"``
        ``"hfs"`` (default): Helton--Far--Speicher contraction via
        :func:`~free_matrix_laws.cauchy_matrix_semicircle`.
        Robust and guaranteed to converge, but slower per step.
        ``"newton"``: Newton's method via :func:`solve_hms`.
        Faster per step but may stall near singularities for
        large-norm $\eta$.
    progress : bool or int
        If True, print a status line approximately every 10% of the
        sweep.  If a positive integer, print every that many steps.
        If False (default), silent.

    Returns
    -------
    dict with keys:
        ``u`` : (m,) array of $u$ values (descending).
        ``tr_W`` : (m,) array of $\operatorname{tr}(W(u))$ (normalized trace).
        ``eigs`` : (m, n) array of eigenvalues of $\Re\,W$ (descending order).
        ``det_W`` : (m,) array of $\det(\Re\,W(u))$.
        ``residuals`` : (m,) array of solver residuals.
        ``converged`` : (m,) bool array.
    """
    from .transforms import cauchy_matrix_semicircle as _hfs_solve

    n = A[0].shape[0] if isinstance(A, (list, tuple)) else A.shape[-1]
    I_n = np.eye(n, dtype=complex)

    # Auto-detect u_max from spectral radius of eta(I)
    if u_max is None:
        etaI = _eta(np.eye(n), A)
        u_max = max(3.0 * la.norm(etaI, 2), 10.0)

    u_vals = np.logspace(np.log10(u_max), np.log10(u_min), n_points)

    tr_W = np.zeros(n_points)
    eigs = np.zeros((n_points, n))
    det_W = np.zeros(n_points)
    residuals = np.zeros(n_points)

    # Progress reporting setup
    if progress is True:
        _prog_every = max(n_points // 10, 1)
    elif progress:
        _prog_every = int(progress)
    else:
        _prog_every = 0

    if _prog_every:
        print(f"  {'step':>5s}/{n_points}  {'u':>10s}  {'tr(W)':>12s}"
              f"  {'residual':>10s}  {'slope':>8s}")
        print("  " + "-" * 58)

    def _report(k):
        """Print progress line with running slope estimate."""
        if not _prog_every:
            return
        if k > 0 and k < n_points - 1 and (k + 1) % _prog_every != 0:
            return
        # Running slope from converged points so far
        slope_str = "..."
        good = residuals[:k+1] < convergence_threshold
        if np.sum(good) >= 5:
            log_u = np.log(u_vals[:k+1][good])
            log_tr = np.log(np.abs(tr_W[:k+1][good]))
            if np.ptp(log_u) > 0.5:  # need some dynamic range
                slope, _ = np.polyfit(log_u, log_tr, 1)
                slope_str = f"{slope:+.4f}"
        conv_char = "✓" if residuals[k] < convergence_threshold else "✗"
        print(f"  {k+1:5d}/{n_points}  {u_vals[k]:10.2e}"
              f"  {tr_W[k]:12.6f}  {residuals[k]:10.2e}"
              f"  {slope_str:>8s}  {conv_char}")

    if method == "newton":
        eta_mat = _build_eta_matrix(A, n)
        W_prev = None
        for k, u in enumerate(u_vals):
            W, iters, res = solve_hms(u, A, W0=W_prev, tol=tol,
                                      maxiter=maxiter, _eta_mat=eta_mat)
            W_prev = W.copy()
            residuals[k] = res
            ReW = 0.5 * (W + W.conj().T)
            ev = np.sort(la.eigvalsh(ReW))[::-1]
            eigs[k] = ev
            tr_W[k] = np.trace(ReW).real / n
            det_W[k] = la.det(ReW).real
            _report(k)
    else:
        # HFS contraction: W(u) = i G(iu)
        G_prev = None
        for k, u in enumerate(u_vals):
            z = 1j * u
            G = _hfs_solve(z, A, G0=G_prev, tol=tol, maxiter=maxiter)
            G_prev = G
            W = 1j * G
            etaW = _eta(W, A)
            residuals[k] = la.norm(etaW @ W + u * W - I_n)
            ReW = 0.5 * (W + W.conj().T)
            ev = np.sort(la.eigvalsh(ReW))[::-1]
            eigs[k] = ev
            tr_W[k] = np.trace(ReW).real / n
            det_W[k] = la.det(ReW).real
            _report(k)

    if _prog_every:
        print()

    return dict(
        u=u_vals,
        tr_W=tr_W,
        eigs=eigs,
        det_W=det_W,
        residuals=residuals,
        converged=residuals < convergence_threshold,
    )


# ═══════════════════════════════════════════════════════════════════════
# Log-log exponent estimation
# ═══════════════════════════════════════════════════════════════════════

def puiseux_exponent(
    u: np.ndarray,
    f: np.ndarray,
    converged: Optional[np.ndarray] = None,
    tail_fraction: float = 0.5,
) -> tuple[float, float, float]:
    r"""
    Estimate Puiseux exponent $\alpha$ from $f(u) \sim C\,u^\alpha$ via
    log-log regression on the smallest $u$ values.

    Parameters
    ----------
    u : (m,) array
        Parameter values (positive, typically descending).
    f : (m,) array
        Function values (positive; absolute value is taken).
    converged : (m,) bool array, optional
        Mask for well-converged points; if given, only these are used.
    tail_fraction : float
        Fraction of (converged) points at the small-$u$ end to use
        for the regression.

    Returns
    -------
    alpha : float
        Estimated exponent.
    C : float
        Estimated prefactor ($e^{\text{intercept}}$).
    r_squared : float
        Coefficient of determination of the log-log fit.
    """
    mask = np.abs(f) > 0
    if converged is not None:
        mask &= converged
    u_use = u[mask]
    f_use = np.abs(f[mask])

    # Sort by u ascending (smallest first)
    order = np.argsort(u_use)
    u_use = u_use[order]
    f_use = f_use[order]

    # Take the tail_fraction smallest-u points
    n_tail = max(int(len(u_use) * tail_fraction), 3)
    u_fit = u_use[:n_tail]
    f_fit = f_use[:n_tail]

    log_u = np.log(u_fit)
    log_f = np.log(f_fit)

    alpha, log_C = np.polyfit(log_u, log_f, 1)
    C = np.exp(log_C)

    # R^2
    ss_res = np.sum((log_f - alpha * log_u - log_C) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return alpha, C, r_sq


# ═══════════════════════════════════════════════════════════════════════
# One-call report
# ═══════════════════════════════════════════════════════════════════════

def singularity_report(
    A,
    u_max: float | None = None,
    u_min: float = 1e-6,
    n_points: int = 300,
    tol: float = 1e-13,
    method: str = "hfs",
    verbose: bool = True,
    progress: bool | int = False,
) -> dict:
    r"""
    Full singularity analysis of the HMS function $W(u)$ at $u = 0$.

    Sweeps $u$ from ``u_max`` to ``u_min``, estimates Puiseux exponents
    for $\operatorname{tr} W(u)$, each eigenvalue of $\Re\,W(u)$, and
    $\det(\Re\,W(u))$.

    Parameters
    ----------
    A : list of (n, n) arrays or (s, n, n) array
        Kraus operators.
    u_max : float or None
        Upper bound for $u$. If None (default), automatically set from
        the spectral radius of $\eta(I)$.
    u_min : float
        Lower bound for $u$.
    n_points : int
        Grid size.
    tol : float
        Solver tolerance.
    method : ``"hfs"`` or ``"newton"``
        Solver to use; see :func:`hms_sweep`.
    verbose : bool
        If True, print a summary table.
    progress : bool or int
        If True, print a progress line approximately every 10%% of the
        sweep, including the current $u$, $\operatorname{tr} W$,
        residual, and a running log-log slope estimate.
        If a positive integer, print every that many steps.

    Returns
    -------
    dict with keys:
        ``sweep`` : dict from :func:`hms_sweep`.
        ``alpha_tr`` : float — exponent for $\operatorname{tr} W$.
        ``C_tr`` : float — prefactor for $\operatorname{tr} W$.
        ``R2_tr`` : float — $R^2$ of the fit for $\operatorname{tr} W$.
        ``alpha_eigs`` : (n,) array — exponents for each eigenvalue.
        ``alpha_det`` : float — exponent for $\det(\Re\,W)$.
        ``singular`` : bool — True if $\operatorname{tr} W(u) \to \infty$.
    """
    sweep = hms_sweep(A, u_max=u_max, u_min=u_min, n_points=n_points,
                      tol=tol, method=method, progress=progress)
    conv = sweep["converged"]
    n_conv = np.sum(conv)

    if n_conv < 10:
        warnings.warn(
            f"Only {n_conv} converged points; results may be unreliable."
        )

    n = sweep["eigs"].shape[1]

    # Trace exponent
    alpha_tr, C_tr, R2_tr = puiseux_exponent(
        sweep["u"], sweep["tr_W"], conv
    )

    # Eigenvalue exponents
    alpha_eigs = np.zeros(n)
    for j in range(n):
        alpha_eigs[j], _, _ = puiseux_exponent(
            sweep["u"], sweep["eigs"][:, j], conv
        )

    # Determinant exponent
    alpha_det, _, R2_det = puiseux_exponent(
        sweep["u"], np.abs(sweep["det_W"]), conv
    )

    # Classification
    singular = alpha_tr < -0.01

    # Density exponent: f(x) ~ |x|^beta where beta = -1 + 2*alpha_eig_max
    # Actually: f(0) = (1/pi) tr Re W(0+), so if tr W ~ u^alpha,
    # and W(u) = i G(iu), density ~ |x|^alpha near 0 (with alpha < 0).
    # More precisely the density exponent equals alpha_tr.

    result = dict(
        sweep=sweep,
        alpha_tr=alpha_tr,
        C_tr=C_tr,
        R2_tr=R2_tr,
        alpha_eigs=alpha_eigs,
        alpha_det=alpha_det,
        R2_det=R2_det,
        singular=singular,
        n_converged=n_conv,
    )

    if verbose:
        u_range = sweep["u"][conv]
        print("=" * 60)
        print("  Singularity report for HMS equation at u = 0")
        print("=" * 60)
        print(f"  Matrix size:          {n} x {n}")
        print(f"  Grid:                 {n_points} points, "
              f"u in [{u_range[-1]:.2e}, {u_range[0]:.2e}]")
        print(f"  Converged:            {n_conv} / {n_points}")
        print()
        if singular:
            print(f"  SINGULAR at u = 0")
            print(f"    tr W(u) ~ {C_tr:.4f} * u^({alpha_tr:.4f})"
                  f"   [R² = {R2_tr:.6f}]")
            # Rational approximation
            from fractions import Fraction
            frac = Fraction(alpha_tr).limit_denominator(12)
            print(f"    Nearest simple fraction: {frac}"
                  f"  (= {float(frac):.6f})")
            print(f"    => density f(x) ~ |x|^({float(frac):.4f})"
                  f" near x = 0")
        else:
            print(f"  NON-SINGULAR at u = 0")
            print(f"    tr W(0+) ≈ {C_tr:.6f}")
            print(f"    => f(0) = {C_tr/np.pi:.6f}")
        print()
        print(f"  Eigenvalue exponents of Re W(u):")
        for j in range(n):
            print(f"    eig_{j+1}:  alpha = {alpha_eigs[j]:.4f}")
        print(f"  det(Re W) exponent:  alpha = {alpha_det:.4f}"
              f"   [R² = {R2_det:.6f}]")
        print("=" * 60)

    return result