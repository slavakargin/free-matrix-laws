
# src/free_matrix_laws/__init__.py
"""
free_matrix_laws: tools for matrix-/operator-valued free probability calculations.
"""

from .opvalued import covariance_map, eta
from .transforms import (
    solve_cauchy_semicircle, solve_G,
    semicircle_density, get_density,
    semicircle_density_scalar
)

__all__ = [
    "covariance_map", "eta",
    "solve_cauchy_semicircle", "solve_G",
    "semicircle_density", "get_density",
    "semicircle_density_scalar"
]

