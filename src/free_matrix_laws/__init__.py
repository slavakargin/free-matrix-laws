
# src/free_matrix_laws/__init__.py
"""
free_matrix_laws: tools for matrix-/operator-valued free probability calculations.
"""

from .opvalued import covariance_map
from .transforms import solve_cauchy_semicircle

# Public aliases
eta = covariance_map
solve_G = solve_cauchy_semicircle

__all__ = [
    "covariance_map", "eta",
    "solve_cauchy_semicircle", "solve_G"
]
