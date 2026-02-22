import numpy as np
import numpy.linalg as la
from free_matrix_laws import covariance_map as eta, cauchy_matrix_semicircle

def test_solver_residual_small():
    rng = np.random.default_rng(0)
    n, s = 3, 3
    A = [0.2*(rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))) for _ in range(s)]
    z = 2.0 + 1.0j
    G = cauchy_matrix_semicircle(z, A, tol=1e-12, maxiter=1000)
    R = z*G - np.eye(n) - eta(G, A) @ G
    assert la.norm(R) < 1e-8

def test_deprecated_aliases_still_work():
    """Old names still importable (they emit DeprecationWarning)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from free_matrix_laws import solve_cauchy_semicircle, solve_G
        rng = np.random.default_rng(1)
        n = 2
        A = [rng.standard_normal((n,n)) for _ in range(2)]
        z = 1.0 + 0.5j
        G1 = solve_cauchy_semicircle(z, A)
        G2 = solve_G(z, A)
        assert np.allclose(G1, G2)