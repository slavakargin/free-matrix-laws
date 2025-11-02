import numpy as np
from free_matrix_laws import semicircle_density as get_density

def semicircle_density_scalar(x, c):
    # scalar semicircle with variance c: support [-2√c, 2√c]
    r = 4.0*c - x*x
    return 0.0 if r <= 0 else (0.5/ (np.pi*c)) * np.sqrt(r)

def test_density_matches_scalar_case_identity_kraus():
    # A_i = σ I ⇒ η(B) = σ^2 B, scalar semicircle with variance c = σ^2
    n = 5
    sigma = 1.5
    c = sigma**2
    A = [sigma * np.eye(n)]
    xs = [-1.0, 0.0, 1.0]
    for x in xs:
        f_num = get_density(x, A, eps=1e-2, tol=1e-12)
        f_ref = semicircle_density_scalar(x, c)
        assert np.isclose(f_num, f_ref, rtol=2e-2, atol=2e-3)

def test_density_accepts_stacked_array():
    n = 4
    sigma = 1.0
    A_stack = np.stack([sigma*np.eye(n)], axis=0)  # (s=1,n,n)
    f1 = get_density(0.0, [sigma*np.eye(n)], eps=1e-2, tol=1e-12)
    f2 = get_density(0.0, A_stack, eps=1e-2, tol=1e-12)
    assert np.isclose(f1, f2, rtol=1e-12, atol=1e-12)
