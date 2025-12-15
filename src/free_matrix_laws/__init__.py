
# src/free_matrix_laws/__init__.py
"""
free_matrix_laws: tools for matrix-/operator-valued free probability calculations.
"""

from .opvalued import (
    covariance_map, eta,
    symmetric_sinkhorn_scale,
    symmetric_sinkhorn_apply,
    symmetric_osi,
    ds_distance,
)
from .transforms import (
    solve_cauchy_semicircle, solve_G,
    semicircle_density, get_density,
    semicircle_density_scalar,
    hfsb_map, solve_cauchy_biased,
    biased_semicircle_density
)

from .ensembles import random_semicircle 

__all__ = [
    "covariance_map", "eta",
    "solve_cauchy_semicircle", "solve_G",
    "semicircle_density", "get_density",
    "semicircle_density_scalar", 
    "random_semicircle",
    "hfsb_map", "solve_cauchy_biased",
    "biased_semicircle_density"
]

