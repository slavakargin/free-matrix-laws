import numpy as np
from free_matrix_laws import matrix_semicircle_density

def semicircle_density_scalar(x, c):
    if abs(x) >= 2.0 * c**0.5:
        return 0.0
    return (1.0 / (2.0 * np.pi * c)) * np.sqrt(4.0 * c - x * x)

def test_scalar_density_recovery():
    n = 3
    c = 1.0
    A = [np.sqrt(c) * np.eye(n)]
    for x in [-1.5, 0.0, 0.3, 1.5]:
        f_num = matrix_semicircle_density(x, A, eps=1e-2, tol=1e-12)
        f_ref = semicircle_density_scalar(x, c)
        assert abs(f_num - f_ref) < 0.05, f"x={x}: {f_num} vs {f_ref}"

def test_stacked_array_input():
    n, sigma = 2, 0.5
    A_stack = np.array([sigma * np.eye(n)])[None, ...].reshape(1, n, n)
    f1 = matrix_semicircle_density(0.0, [sigma*np.eye(n)], eps=1e-2, tol=1e-12)
    f2 = matrix_semicircle_density(0.0, A_stack, eps=1e-2, tol=1e-12)
    assert abs(f1 - f2) < 1e-8