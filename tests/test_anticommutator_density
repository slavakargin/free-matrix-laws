import numpy as np
import numpy.linalg as la

from free_matrix_laws.transforms import solve_cauchy_linearized, polynomial_density


def _anticommutator_data():
    a0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    a1 = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    a2 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    A_list = [a1, a2]
    A_stack = np.stack(A_list, axis=0)
    return a0, A_list, A_stack


def test_density_matches_corner_imag_part():
    a0, A_list, _ = _anticommutator_data()

    x = 0.40
    eps = 2e-2
    z = x + 1j * eps

    # Solve for G(z, b_eps(z)) and extract the (1,1) corner
    G = solve_cauchy_linearized(
        z, a0, A_list,
        eps_reg=eps,       # use same small reg for this test
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )
    m = G[0, 0]  # block_size=1 => scalar corner
    f_from_G = (-1.0 / np.pi) * np.imag(m)

    # Compare with the packaged density helper
    f = polynomial_density(
        x, a0, A_list,
        eps=eps,
        eps_reg=eps,
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )

    assert np.isfinite(f)
    assert abs(f - f_from_G) <= 1e-10


def test_list_vs_stack_agree():
    a0, A_list, A_stack = _anticommutator_data()

    x = 0.75
    eps = 3e-2

    f_list = polynomial_density(
        x, a0, A_list,
        eps=eps,
        eps_reg=eps,
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )
    f_stack = polynomial_density(
        x, a0, A_stack,
        eps=eps,
        eps_reg=eps,
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )

    assert np.isfinite(f_list)
    assert np.isfinite(f_stack)
    assert abs(f_list - f_stack) <= 1e-10


def test_density_is_approximately_even():
    # For p(X1,X2)=X1X2+X2X1 the law is symmetric, so density should be even.
    # Numerically (finite eps) we check only approximate symmetry.
    a0, A_list, _ = _anticommutator_data()

    x = 0.60
    eps = 5e-2  # slightly larger eps -> more stable symmetry check

    fp = polynomial_density(
        x, a0, A_list,
        eps=eps,
        eps_reg=eps,
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )
    fm = polynomial_density(
        -x, a0, A_list,
        eps=eps,
        eps_reg=eps,
        block_size=1,
        tol=1e-12,
        maxiter=20_000,
    )

    # Expect only a modest tolerance because eps regularizes.
    assert abs(fp - fm) <= 5e-3
    # Also: density should be nonnegative up to a small numerical wiggle
    assert fp >= -5e-3
    assert fm >= -5e-3
