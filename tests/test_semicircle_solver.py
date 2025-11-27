import numpy as np
import numpy.linalg as la
from free_matrix_laws import covariance_map as eta, solve_cauchy_semicircle, solve_G

def test_solver_residual_small():
    rng = np.random.default_rng(0)
    n, s = 3, 3
    A = [0.2*(rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))) for _ in range(s)]
    z = 2.0 + 1.0j
    G = solve_cauchy_semicircle(z, A, tol=1e-12, maxiter=1000)
    R = z*G - np.eye(n) - eta(G, A) @ G
    assert la.norm(R) < 1e-8

def test_solver_alias_is_same_object():
    assert solve_G is solve_cauchy_semicircle