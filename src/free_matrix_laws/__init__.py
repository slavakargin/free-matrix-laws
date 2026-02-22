# src/free_matrix_laws/__init__.py
"""
free_matrix_laws: tools for matrix-/operator-valued free probability calculations.
"""

__version__ = "0.1.0"

from .opvalued import (
    covariance_map, eta,
    symmetric_sinkhorn_scale,
    symmetric_sinkhorn_apply,
    symmetric_osi,
    ds_distance,
)

# ── primary API (v0.1.0) ────────────────────────────────────────────────
from .transforms import (
    # Cauchy transforms (matrix-valued)
    cauchy_matrix_semicircle,
    cauchy_kronecker_semicircle,
    h_kronecker_semicircle,
    cauchy_biased_matrix_semicircle,
    cauchy_polynomial,
    # Scalar densities
    matrix_semicircle_density,
    biased_matrix_semicircle_density,
    polynomial_density,
    # Scalar helpers
    semicircle_density_scalar,
    semicircle_cauchy_scalar,
    # Utilities
    lambda_eps,
)

from .ensembles import random_semicircle

from .quadrature import (
    cauchy_matrix_semicircle_bruteforce,
    h_matrix_semicircle_bruteforce,
    G_from_h,
    density_scalar_quadrature,
)

# ── deprecated aliases (will warn) ──────────────────────────────────────
from .transforms import (
    solve_cauchy_semicircle,       # → cauchy_matrix_semicircle
    solve_G,                       # → cauchy_matrix_semicircle
    semicircle_density,            # → matrix_semicircle_density
    get_density,                   # → matrix_semicircle_density
    solve_cauchy_biased,           # → cauchy_biased_matrix_semicircle
    biased_semicircle_density,     # → biased_matrix_semicircle_density
    solve_cauchy_linearized,       # → cauchy_polynomial
    polynomial_semicircle_density, # → polynomial_density
    hfsb_map,                      # → _hfsb_map (now private)
)

__all__ = [
    # opvalued
    "covariance_map", "eta",
    "symmetric_sinkhorn_scale",
    "symmetric_sinkhorn_apply",
    "symmetric_osi",
    "ds_distance",
    # transforms — primary
    "cauchy_matrix_semicircle",
    "cauchy_kronecker_semicircle",
    "h_kronecker_semicircle",
    "cauchy_biased_matrix_semicircle",
    "cauchy_polynomial",
    "matrix_semicircle_density",
    "biased_matrix_semicircle_density",
    "polynomial_density",
    "semicircle_density_scalar",
    "semicircle_cauchy_scalar",
    "lambda_eps",
    # ensembles
    "random_semicircle",
    # quadrature
    "cauchy_matrix_semicircle_bruteforce",
    "h_matrix_semicircle_bruteforce",
    "G_from_h",
    "density_scalar_quadrature",
    # deprecated (still importable, will warn)
    "solve_cauchy_semicircle", "solve_G",
    "semicircle_density", "get_density",
    "solve_cauchy_biased",
    "biased_semicircle_density",
    "solve_cauchy_linearized",
    "polynomial_semicircle_density",
    "hfsb_map",
]