import numpy as np
from numpy.linalg import norm
from free_matrix_laws import random_semicircle

def test_real_is_symmetric_and_scaling():
    n, c = 200, 1.7
    H = random_semicircle(n, field="real", variance=c, seed=0)
    assert np.allclose(H, H.T, atol=1e-12)
    # off-diagonal variance ~ c/n (empirical, up to constants)
    idx = np.triu_indices(n, 1)
    v_emp = np.var(H[idx])
    assert np.isclose(v_emp, c/n, rtol=0.35)  # loose, finite-n

def test_complex_is_hermitian_and_scaling():
    n, c = 200, 0.8
    H = random_semicircle(n, field="complex", variance=c, seed=1)
    assert np.allclose(H, H.conj().T, atol=1e-12)
    idx = np.triu_indices(n, 1)
    v_emp = np.mean(np.abs(H[idx])**2)
    assert np.isclose(v_emp, c/n, rtol=0.35)

def test_spectral_radius_matches_variance_coarsely():
    n, c = 300, 1.0
    H = random_semicircle(n, field="real", variance=c, seed=2)
    # ||H|| ~ 2*sqrt(c) as n grows; allow slack for small n
    smax = np.linalg.svd(H, compute_uv=False)[0]
    assert 1.5 <= smax <= 2.7
