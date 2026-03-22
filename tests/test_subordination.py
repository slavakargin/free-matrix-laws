"""Tests for subordination_kronecker."""
import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws import (
    subordination_kronecker,
    cauchy_kronecker_semicircle,
    cauchy_kronecker,
    h_kronecker,
    lambda_eps,
    semicircle_cauchy_scalar,
    polynomial_density,
)


# Standard anticommutator linearization data
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

# Default eps used by subordination_kronecker for rank-deficient matrices
SUB_EPS = 1e-4


class TestSubordinationKroneckerBasic:
    def test_returns_correct_shape(self):
        n = 3
        z = 0.5 + 0.01j
        b = lambda_eps(z, n) - A0
        omega = subordination_kronecker(b, A1, A2)
        assert omega.shape == (n, n)
        assert omega.dtype == complex

    def test_convergence_info(self):
        z = 0.5 + 0.01j
        b = lambda_eps(z, 3) - A0
        omega, info = subordination_kronecker(b, A1, A2, return_info=True)
        assert "iters" in info
        assert "last_diff" in info
        assert info["last_diff"] < 1e-7

    def test_fixed_point_property(self):
        r"""omega should satisfy omega = h_Y(h_X(omega) + b) + b
        when using the same eps as the subordination solver."""
        z = 0.5 + 0.05j
        b = lambda_eps(z, 3) - A0
        omega = subordination_kronecker(b, A1, A2, eps=SUB_EPS)

        # Must use same eps as the solver to verify fixed point
        W1 = h_kronecker(omega, A1, eps=SUB_EPS) + b
        W2 = h_kronecker(W1, A2, eps=SUB_EPS) + b

        assert np.allclose(omega, W2, atol=1e-6), (
            f"Fixed-point residual: {la.norm(omega - W2):.2e}"
        )


class TestSubordinationVsLinearization:
    r"""Cross-check: subordination density should match polynomial_density
    for the anticommutator XY + YX of two semicircles."""

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, -1.5])
    def test_density_matches_polynomial_solver(self, x):
        eps_im = 0.01
        z = x + eps_im * 1j
        n = 3
        b = lambda_eps(z, n) - A0

        # Subordination approach
        omega = subordination_kronecker(b, A1, A2)
        G = cauchy_kronecker(omega, A1, eps=SUB_EPS)
        f_sub = -G[0, 0].imag / np.pi

        # Linearization (fixed-point) approach
        f_lin = polynomial_density(x, A0, np.array([A1, A2]), eps=eps_im,
                                   block_size=1, tol=1e-10, maxiter=10000)

        assert abs(f_sub - f_lin) < 0.05, (
            f"x={x}: subordination={f_sub:.6f}, linearization={f_lin:.6f}"
        )


class TestSubordinationCustomTransform:
    """Test with non-semicircle scalar transforms."""

    def test_fixed_point_with_custom(self):
        """Fixed-point property should hold for a custom scalar transform.
        Use scaled semicircle (variance c) as a non-default transform."""
        def cauchy_scaled_sc(z, c=2.0):
            """Semicircle with variance c."""
            z = np.asarray(z, dtype=complex)
            disc = np.sqrt(z**2 - 4.0 * c)
            disc = np.where(disc.imag * z.imag < 0, -disc, disc)
            return (z - disc) / (2.0 * c)

        z = 0.5 + 0.05j
        b = lambda_eps(z, 3) - A0
        eps = SUB_EPS
        omega = subordination_kronecker(
            b, A1, A2,
            cauchy_scalar_x=cauchy_scaled_sc,
            cauchy_scalar_y=cauchy_scaled_sc,
            eps=eps,
        )

        W1 = h_kronecker(omega, A1, cauchy_scalar=cauchy_scaled_sc, eps=eps) + b
        W2 = h_kronecker(W1, A2, cauchy_scalar=cauchy_scaled_sc, eps=eps) + b

        assert np.allclose(omega, W2, atol=1e-6), (
            f"Fixed-point residual: {la.norm(omega - W2):.2e}"
        )

    def test_mixed_distributions(self):
        """X semicircle, Y free Poisson -- fixed-point should still hold."""
        def cauchy_poisson(z, lam=4.0):
            z = np.asarray(z, dtype=complex)
            a = (1 - np.sqrt(lam))**2
            b_val = (1 + np.sqrt(lam))**2
            disc = np.sqrt((z - a) * (z - b_val))
            disc = np.where(disc.imag * z.imag < 0, -disc, disc)
            return (1 + z - lam - disc) / (2 * z)

        z = 5.0 + 0.1j
        b = lambda_eps(z, 3) - A0
        eps = 1e-4

        omega = subordination_kronecker(
            b, A1, A2,
            cauchy_scalar_x=semicircle_cauchy_scalar,
            cauchy_scalar_y=cauchy_poisson,
            eps=eps,
        )

        W1 = h_kronecker(omega, A1, cauchy_scalar=semicircle_cauchy_scalar, eps=eps) + b
        W2 = h_kronecker(W1, A2, cauchy_scalar=cauchy_poisson, eps=eps) + b

        assert np.allclose(omega, W2, atol=1e-6), (
            f"Fixed-point residual: {la.norm(omega - W2):.2e}"
        )