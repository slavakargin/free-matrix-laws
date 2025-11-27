import numpy as np
import numpy.linalg as la

from free_matrix_laws import (
    covariance_map as eta,
    solve_cauchy_semicircle,
    solve_cauchy_biased,
    hfsb_map
)

def test_biased_reduces_to_unbiased_when_a0_zero():
    rng = np.random.default_rng(0)
    n, s = 5, 3
    A = [rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)) for _ in range(s)]
    a0 = np.zeros((n,n), dtype=complex)
    z = 0.2 + 1j*0.03

    G_unb = solve_cauchy_semicircle(z, A, tol=1e-12, maxiter=4000)
    G_bi, info = solve_cauchy_biased(z, a0, A, tol=1e-12, maxiter=4000, return_info=True)
    assert la.norm(G_unb - G_bi, 'fro') <= 5e-8
    assert info["residual"] <= 1e-10

def test_hfsb_residual_decreases_a_few_steps():
    rng = np.random.default_rng(1)
    n, s = 4, 2
    a0 = rng.standard_normal((n,n))  # real bias OK
    A = [rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)) for _ in range(s)]
    z = -0.4 + 1j*0.05

    I = np.eye(n)
    G = 1j*I/np.imag(z)
    def residual(G):
        return la.norm((z*I - a0)@G - I - eta(G, A)@G, 'fro')

    r0 = residual(G)
    for _ in range(5):
        G = hfsb_map(G, z, a0, A, relax=0.5)
    r1 = residual(G)
    assert r1 < r0
