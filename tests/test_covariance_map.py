import numpy as np
from free_matrix_laws import covariance_map, eta

def test_covariance_map_matches_kraus_loop_nonhermitian():
    rng = np.random.default_rng(0)
    n, s = 4, 3
    A = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) for _ in range(s)]
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    out_vec = covariance_map(B, A)
    out_loop = sum(Ai @ B @ Ai.conj().T for Ai in A)
    assert np.allclose(out_vec, out_loop)

def test_accepts_stacked_array_equivalence():
    rng = np.random.default_rng(1)
    n, s = 3, 2
    A_list = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) for _ in range(s)]
    A_stack = np.stack(A_list, axis=0)
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    assert np.allclose(covariance_map(B, A_list), covariance_map(B, A_stack))

def test_eta_alias():
    rng = np.random.default_rng(2)
    n, s = 2, 2
    A = [rng.standard_normal((n, n)) for _ in range(s)]
    B = rng.standard_normal((n, n))
    assert np.allclose(eta(B, A), covariance_map(B, A))
