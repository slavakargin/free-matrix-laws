import numpy as np
from numpy import linalg as la
from free_matrix_laws import semicircle_density, biased_semicircle_density

def rand_A(n, s, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)) for _ in range(s)]

def test_biased_matches_unbiased_when_a0_zero():
    n, s = 4, 3
    A = rand_A(n, s, seed=1)
    a0 = np.zeros((n,n), dtype=complex)
    x = 0.2
    f_unb = semicircle_density(x, A, eps=1e-2, tol=1e-11, maxiter=4000)
    f_bia = biased_semicircle_density(x, a0, A, eps=1e-2, tol=1e-11, maxiter=4000)
    assert np.allclose(f_unb, f_bia, rtol=1e-6, atol=1e-6)

def test_biased_rejects_wrong_shape_a0():
    n, s = 3, 2
    A = rand_A(n, s, seed=2)
    bad = np.zeros((n+1, n+1))
    try:
        _ = biased_semicircle_density(0.0, bad, A)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for mismatched a0 shape"
