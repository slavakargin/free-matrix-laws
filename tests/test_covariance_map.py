# tests/test_covariance_map.py
import numpy as np
from free_matrix_laws import covariance_map

def test_covariance_map_matches_loop():
    rng = np.random.default_rng(0)
    n, s = 3, 4
    A = [rng.standard_normal((n,n)) for _ in range(s)]
    # make them Hermitian for the usual setting
    A = [0.5*(Ai + Ai.T) for Ai in A]
    B = rng.standard_normal((n,n))

    out_vec = covariance_map(B, A)
    out_loop = sum(Ai @ B @ Ai for Ai in A)
    assert np.allclose(out_vec, out_loop)
