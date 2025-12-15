import numpy as np
import numpy.linalg as la
from free_matrix_laws import (
    symmetric_sinkhorn_scale, symmetric_sinkhorn_apply,
    symmetric_osi, ds_distance
)

def random_kraus(n=5, s=3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((s,n,n)) + 1j*rng.standard_normal((s,n,n))
    return A

def test_scale_reduces_ds_distance():
    A = random_kraus()
    ds0 = ds_distance(A)
    B = symmetric_sinkhorn_scale(A)
    ds1 = ds_distance(B)
    assert ds1 < ds0

def test_apply_matches_explicit_sum():
    n = 4
    A = random_kraus(n=n, s=2, seed=1)
    X = np.eye(n)
    Y1 = symmetric_sinkhorn_apply(X, A)
    B = symmetric_sinkhorn_scale(A, preserve_input_type=False)
    Y2 = sum(Bi @ X @ Bi.conj().T for Bi in B)
    assert np.allclose(Y1, Y2)

def test_osi_converges_reasonably():
    A = random_kraus(n=4, s=2, seed=2)
    B, info = symmetric_osi(A, maxiter=30, tol=1e-8, return_history=True)
    assert info["ds"] <= 1e-8 or info["iters"] == 30
