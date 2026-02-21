r"""
Brute-force quadrature for the matrix-valued Cauchy transform of $b \otimes X$.

This module provides a direct numerical-integration approach to computing
the operator-valued Cauchy transform

$$
    G_b(w) \;=\; \int_{-\infty}^{\infty}
        \frac{1}{w - x\,b}\;\mathrm{d}\mu_{\mathrm{sc}}(x),
$$

where $b$ is a deterministic $n \times n$ matrix, $X$ is a standard
semicircular random variable, and $\mu_{\mathrm{sc}}$ denotes the Wigner
semicircle law with density
$\rho(x)=\frac{1}{2\pi}\sqrt{4-x^2}\,\mathbf{1}_{|x|\le 2}$.

The resolvent $(w - x\,b)^{-1}$ is integrated entry-by-entry via
adaptive Gauss--Kronrod quadrature (``scipy.integrate.quad``), with the
integration contour shifted slightly into the upper half-plane
($x \mapsto x + i\varepsilon$) to regularize the pole on the real axis.

The companion function :func:`h_matrix_semicircle` computes the
subordination-style "h-function"
$$
    h_b(w) \;=\; G_b(w)^{-1} - w,
$$
which is related to the $R$-transform in operator-valued free probability.

.. note::

   This is a **brute-force** reference implementation.  For production
   use on large matrices or fine grids, prefer the fixed-point solvers
   in :mod:`free_matrix_laws.transforms` (e.g.
   :func:`~free_matrix_laws.transforms.solve_cauchy_semicircle`), which
   are typically orders of magnitude faster.  The quadrature approach is
   most useful for:

   * **Testing / validation** — cross-checking the fixed-point solvers
     against a completely independent method.
   * **Small one-off evaluations** where implementation simplicity
     matters more than speed.

References
----------
* J. W. Helton, R. Rashidi Far, R. Speicher,
  *Operator-valued Semicircular Elements*, IMRN (2007).
* R. Speicher, *Combinatorial Theory of the Free Product with
  Amalgamation and Operator-Valued Free Probability Theory*,
  Memoirs AMS **132** (1998).
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
import numpy.linalg as la

from .transforms import semicircle_cauchy_scalar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_square(M: np.ndarray, name: str = "M") -> int:
    """Check that *M* is a 2-d square array and return its size *n*."""
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square (n, n); got shape {M.shape!r}")
    return M.shape[0]


def _ensure_complex_matrix(w: np.ndarray, n: int) -> np.ndarray:
    r"""Promote *w* to a complex ``(n, n)`` array."""
    w = np.asarray(w, dtype=np.complex128)
    if w.shape != (n, n):
        raise ValueError(
            f"w must have shape ({n}, {n}); got {w.shape!r}"
        )
    return w


# ---------------------------------------------------------------------------
# Core: matrix-valued integrand
# ---------------------------------------------------------------------------

def _matrix_integrand(
    x: float,
    w: np.ndarray,
    b: np.ndarray,
    eps: float,
) -> np.ndarray:
    r"""
    Evaluate the matrix-valued integrand at a single real point *x*.

    $$
        f(x) \;=\; -\frac{1}{\pi}\,
        (w - z\,b)^{-1}\,\operatorname{Im}\!\big[G_{\mathrm{sc}}(z)\big],
        \qquad z = x + i\varepsilon,
    $$

    where $G_{\mathrm{sc}}$ is the scalar semicircle Cauchy transform.
    """
    z = complex(x + 1j * eps)
    g_sc = semicircle_cauchy_scalar(z)
    resolvent = la.inv(w - z * b)
    return -(1.0 / np.pi) * resolvent * np.imag(g_sc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cauchy_matrix_semicircle(
    w: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1e-3,
    x_min: float = -4.0,
    x_max: float = 4.0,
    quad_opts: Optional[dict] = None,
) -> np.ndarray:
    r"""
    Matrix-valued Cauchy (Stieltjes) transform of $b \otimes X$ by quadrature.

    Computes the $n \times n$ matrix

    $$
        G_b(w) \;=\; \int_{x_{\min}}^{x_{\max}}
            -\frac{1}{\pi}\,(w - z\,b)^{-1}\,
            \operatorname{Im}\!\big[G_{\mathrm{sc}}(z)\big]\,\mathrm{d}x,
        \qquad z = x + i\varepsilon,
    $$

    where $G_{\mathrm{sc}}(z)=\frac{z-\sqrt{z^2-4}}{2}$ is the scalar
    semicircle Cauchy transform.

    Each matrix entry is integrated independently with
    ``scipy.integrate.quad``.

    Parameters
    ----------
    w : (n, n) array_like, complex
        The matrix spectral parameter.  Typically
        $w = (\alpha + i\delta)\,I_n$ for some small $\delta > 0$.
    b : (n, n) array_like
        Deterministic coefficient matrix.
    eps : float, default ``1e-3``
        Imaginary shift $\varepsilon > 0$ applied to the integration
        variable ($z = x + i\varepsilon$).  Controls regularization of the
        resolvent near the real axis.
    x_min, x_max : float, defaults ``-4.0``, ``4.0``
        Integration limits.  The semicircle density is supported on
        $[-2, 2]$, so any limits beyond that suffice; wider limits have
        negligible cost thanks to adaptive quadrature.
    quad_opts : dict, optional
        Extra keyword arguments forwarded to ``scipy.integrate.quad``
        (e.g. ``{'limit': 100, 'epsabs': 1e-10}``).

    Returns
    -------
    G : (n, n) ndarray, complex
        The matrix-valued Cauchy transform $G_b(w)$.

    Raises
    ------
    ValueError
        If *w* or *b* are not square matrices of compatible shapes.

    Notes
    -----
    * For the integration to be well-defined the imaginary shift
      $\varepsilon$ must be strictly positive.
    * The integration limits should cover $[-2, 2]$ (support of the
      semicircle law).  The default $[-4, 4]$ is generous.
    * Complexity is $O(n^3 \cdot Q)$ per matrix entry, where $Q$ is the
      number of quadrature nodes chosen by ``quad``.  Total cost thus
      scales as $O(n^5\, Q)$, which is acceptable for small $n$.

    See Also
    --------
    free_matrix_laws.transforms.solve_cauchy_semicircle :
        Much faster fixed-point solver (preferred for production).
    h_matrix_semicircle :
        The companion "h-function" $h_b(w) = G_b(w)^{-1} - w$.

    Examples
    --------
    >>> import numpy as np
    >>> from free_matrix_laws.quadrature import cauchy_matrix_semicircle
    >>> b = np.array([[1, 0], [0, 0.5]])
    >>> w = (0.5 + 0.1j) * np.eye(2)
    >>> G = cauchy_matrix_semicircle(w, b, eps=1e-3)
    >>> G.shape
    (2, 2)
    """
    try:
        from scipy.integrate import quad
    except ImportError as exc:
        raise ImportError(
            "cauchy_matrix_semicircle requires scipy.  "
            "Install it with:  pip install scipy"
        ) from exc

    if eps <= 0:
        raise ValueError(f"eps must be > 0; got {eps}")
    if x_min >= x_max:
        raise ValueError(f"Require x_min < x_max; got [{x_min}, {x_max}]")

    b = np.asarray(b, dtype=np.complex128)
    n = _validate_square(b, "b")
    w = _ensure_complex_matrix(w, n)

    quad_kw = dict(quad_opts) if quad_opts is not None else {}

    # Determine the matrix shape from a probe evaluation
    result = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            # Scalar function for the (i, j) entry
            def _scalar_re(x, _w=w, _b=b, _eps=eps, _i=i, _j=j):
                return _matrix_integrand(x, _w, _b, _eps)[_i, _j].real

            def _scalar_im(x, _w=w, _b=b, _eps=eps, _i=i, _j=j):
                return _matrix_integrand(x, _w, _b, _eps)[_i, _j].imag

            re_part, _ = quad(_scalar_re, x_min, x_max, **quad_kw)
            im_part, _ = quad(_scalar_im, x_min, x_max, **quad_kw)
            result[i, j] = re_part + 1j * im_part

    return result


def h_matrix_semicircle(
    w: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1e-3,
    x_min: float = -4.0,
    x_max: float = 4.0,
    quad_opts: Optional[dict] = None,
) -> np.ndarray:
    r"""
    Subordination "h-function" for the matrix semicircle via quadrature.

    Computes
    $$
        h_b(w) \;=\; G_b(w)^{-1} \;-\; w,
    $$
    where $G_b(w)$ is the matrix-valued Cauchy transform of $b \otimes X$
    obtained by :func:`cauchy_matrix_semicircle`.

    Parameters
    ----------
    w : (n, n) array_like, complex
        Matrix spectral parameter.
    b : (n, n) array_like
        Deterministic coefficient matrix.
    eps : float, default ``1e-3``
        Imaginary shift for regularization (passed to
        :func:`cauchy_matrix_semicircle`).
    x_min, x_max : float
        Integration limits (passed through).
    quad_opts : dict, optional
        Extra keywords for ``scipy.integrate.quad``.

    Returns
    -------
    h : (n, n) ndarray, complex
        The matrix $h_b(w)$.

    See Also
    --------
    cauchy_matrix_semicircle : The underlying Cauchy-transform computation.

    Examples
    --------
    >>> import numpy as np
    >>> from free_matrix_laws.quadrature import h_matrix_semicircle
    >>> b = np.eye(3)
    >>> w = (0.5 + 0.05j) * np.eye(3)
    >>> h = h_matrix_semicircle(w, b, eps=1e-3)
    >>> h.shape
    (3, 3)
    """
    w = np.asarray(w, dtype=np.complex128)
    G = cauchy_matrix_semicircle(
        w, b, eps=eps, x_min=x_min, x_max=x_max, quad_opts=quad_opts
    )
    return la.inv(G) - w


def G_from_h(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    r"""
    Recover the Cauchy transform $G_b(w)$ from the h-function.

    Given the relation
    $$
        h_b(w) \;=\; G_b(w)^{-1} - w,
    $$
    this inverts it to obtain
    $$
        G_b(w) \;=\; \bigl(h_b(w) + w\bigr)^{-1}.
    $$

    Parameters
    ----------
    h : (n, n) array_like, complex
        The h-function value $h_b(w)$.
    w : (n, n) array_like, complex
        The matrix spectral parameter at which $h$ was evaluated.

    Returns
    -------
    G : (n, n) ndarray, complex
        The matrix-valued Cauchy transform $G_b(w)$.

    See Also
    --------
    h_matrix_semicircle :
        Computes $h_b(w)$ from $w$ and $b$ via quadrature.
    cauchy_matrix_semicircle :
        Computes $G_b(w)$ directly via quadrature.

    Examples
    --------
    >>> import numpy as np
    >>> from free_matrix_laws.quadrature import (
    ...     cauchy_matrix_semicircle, h_matrix_semicircle, G_from_h,
    ... )
    >>> b = np.eye(2)
    >>> w = (0.5 + 0.1j) * np.eye(2)
    >>> h = h_matrix_semicircle(w, b, eps=1e-3)
    >>> G_recovered = G_from_h(h, w)
    >>> G_direct    = cauchy_matrix_semicircle(w, b, eps=1e-3)
    >>> np.allclose(G_recovered, G_direct)
    True
    """
    h = np.asarray(h, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    if h.shape != w.shape:
        raise ValueError(
            f"h and w must have the same shape; got {h.shape!r} vs {w.shape!r}"
        )
    return la.inv(h + w)


def density_scalar_quadrature(
    x: float,
    b: np.ndarray,
    *,
    eps: float = 1e-3,
    x_min: float = -4.0,
    x_max: float = 4.0,
    quad_opts: Optional[dict] = None,
) -> float:
    r"""
    Scalar density of $b \otimes X$ at a real point, via quadrature.

    Evaluates $G_b(w)$ at $w = (x + i\varepsilon)\,I_n$ and returns the
    Stieltjes-inversion density

    $$
        f(x) \;=\; -\frac{1}{\pi}\,\operatorname{Im}\!\left(
            \frac{1}{n}\,\operatorname{tr}\, G_b\!\big((x+i\varepsilon)\,I_n\big)
        \right).
    $$

    Parameters
    ----------
    x : float
        Real evaluation point.
    b : (n, n) array_like
        Deterministic coefficient matrix.
    eps : float, default ``1e-3``
        Imaginary shift $\varepsilon > 0$.
    x_min, x_max : float
        Integration limits.
    quad_opts : dict, optional
        Extra keywords for ``scipy.integrate.quad``.

    Returns
    -------
    float
        Approximate density $f(x) \ge 0$.

    Examples
    --------
    >>> import numpy as np
    >>> from free_matrix_laws.quadrature import density_scalar_quadrature
    >>> b = np.eye(2)
    >>> density_scalar_quadrature(0.0, b, eps=1e-2)  # doctest: +SKIP
    0.318...
    """
    b = np.asarray(b, dtype=np.complex128)
    n = _validate_square(b, "b")
    w = complex(x + 1j * eps) * np.eye(n, dtype=np.complex128)

    # The integration regularization eps_quad should be independent of
    # the spectral-parameter imaginary part.  A safe default is 1e-3,
    # but when the caller's eps is already large we can use a smaller
    # integration shift.
    eps_quad = min(eps, 1e-3)

    G = cauchy_matrix_semicircle(
        w, b, eps=eps_quad, x_min=x_min, x_max=x_max, quad_opts=quad_opts
    )
    m = np.trace(G) / n
    return float(-(1.0 / np.pi) * np.imag(m))
