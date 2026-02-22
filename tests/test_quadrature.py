"""
Tests for ``free_matrix_laws.quadrature``
— brute-force integration of the matrix-valued Cauchy transform.
"""
import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws.quadrature import (
    cauchy_matrix_semicircle_bruteforce,
    h_matrix_semicircle_bruteforce,
    G_from_h,
    density_scalar_quadrature,
)
from free_matrix_laws.transforms import semicircle_cauchy_scalar


# ── helpers ──────────────────────────────────────────────────────────────
def _scalar_eye_G(z):
    """When b = I, G_b(w) = G_sc(z) * I  with w = z*I."""
    return complex(semicircle_cauchy_scalar(z))


# ── shape / type tests ───────────────────────────────────────────────────

class TestCauchyMatrixSemicircleShapes:
    def test_returns_square_matrix(self):
        n = 3
        b = np.eye(n)
        w = (0.5 + 0.1j) * np.eye(n)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        assert G.shape == (n, n)
        assert np.iscomplexobj(G)

    def test_2x2(self):
        b = np.array([[1.0, 0.0], [0.0, 0.5]])
        w = (0.5 + 0.1j) * np.eye(2)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        assert G.shape == (2, 2)

    def test_1x1(self):
        b = np.array([[2.0]])
        w = np.array([[0.3 + 0.2j]])
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        assert G.shape == (1, 1)


# ── validation / error tests ────────────────────────────────────────────

class TestCauchyMatrixSemicircleValidation:
    def test_non_square_b_raises(self):
        with pytest.raises(ValueError, match="square"):
            cauchy_matrix_semicircle_bruteforce(
                np.eye(2, dtype=complex), np.ones((2, 3)), eps=1e-3
            )

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="shape"):
            cauchy_matrix_semicircle_bruteforce(
                np.eye(2, dtype=complex), np.eye(3), eps=1e-3
            )

    def test_eps_nonpositive_raises(self):
        with pytest.raises(ValueError, match="eps"):
            cauchy_matrix_semicircle_bruteforce(np.eye(2, dtype=complex), np.eye(2), eps=0)

    def test_bad_integration_limits_raises(self):
        with pytest.raises(ValueError, match="x_min"):
            cauchy_matrix_semicircle_bruteforce(
                np.eye(2, dtype=complex), np.eye(2), eps=1e-3, x_min=5, x_max=2
            )


# ── identity b tests (b = I) ────────────────────────────────────────────

class TestIdentityB:
    """When b = I the matrix Cauchy transform G_b(w) = G_sc(z)*I."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_diagonal_matches_scalar(self, n):
        z = 0.5 + 0.1j
        w = z * np.eye(n, dtype=complex)
        b = np.eye(n, dtype=float)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        g_scalar = _scalar_eye_G(z)
        # diagonal entries should match the scalar Cauchy transform
        for k in range(n):
            assert abs(G[k, k] - g_scalar) < 2e-3, (
                f"G[{k},{k}]={G[k,k]}, expected ~{g_scalar}"
            )

    @pytest.mark.parametrize("n", [2, 3])
    def test_off_diagonal_near_zero(self, n):
        z = 0.5 + 0.1j
        w = z * np.eye(n, dtype=complex)
        b = np.eye(n, dtype=float)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(G[i, j]) < 1e-6, (
                        f"G[{i},{j}]={G[i,j]}, expected ~0"
                    )


# ── h-function tests ────────────────────────────────────────────────────

class TestHMatrixSemicircle:
    def test_returns_correct_shape(self):
        n = 2
        b = np.eye(n)
        w = (0.5 + 0.1j) * np.eye(n)
        h = h_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        assert h.shape == (n, n)
        assert np.iscomplexobj(h)

    def test_identity_b_consistency(self):
        """For b = I: h(w) = G(w)^{-1} - w, and the scalar version
        satisfies h_sc(z) = g_sc(z)^{-1} - z."""
        z = 0.5 + 0.1j
        n = 2
        w = z * np.eye(n, dtype=complex)
        b = np.eye(n, dtype=float)
        h = h_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        g_sc = _scalar_eye_G(z)
        h_scalar = 1.0 / g_sc - z
        for k in range(n):
            assert abs(h[k, k] - h_scalar) < 5e-3


# ── G_from_h (inverse of h) tests ────────────────────────────────────────

class TestGFromH:
    """G_from_h should invert h_matrix_semicircle_bruteforce exactly."""

    def test_roundtrip_identity_b(self):
        """G -> h -> G roundtrip for b = I."""
        n = 2
        b = np.eye(n)
        w = (0.5 + 0.1j) * np.eye(n, dtype=complex)
        G_direct = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        h = h_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        G_recovered = G_from_h(h, w)
        assert np.allclose(G_direct, G_recovered, atol=1e-12)

    def test_roundtrip_general_b(self):
        """G -> h -> G roundtrip for a non-trivial b."""
        b = np.array([[1.0, 0.3], [0.3, 0.7]])
        w = (0.4 + 0.2j) * np.eye(2, dtype=complex)
        G_direct = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        h = h_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        G_recovered = G_from_h(h, w)
        assert np.allclose(G_direct, G_recovered, atol=1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            G_from_h(np.eye(2, dtype=complex), np.eye(3, dtype=complex))

    def test_returns_correct_shape(self):
        h = (-0.3 + 0.2j) * np.eye(3, dtype=complex)
        w = (0.5 + 0.1j) * np.eye(3, dtype=complex)
        G = G_from_h(h, w)
        assert G.shape == (3, 3)
        assert np.iscomplexobj(G)


# ── density recovery tests ──────────────────────────────────────────────

class TestDensityScalarQuadrature:
    def test_identity_b_center(self):
        """density(0, I) ≈ 1/π (standard semicircle peak)."""
        b = np.eye(2)
        d = density_scalar_quadrature(0.0, b, eps=1e-2)
        assert abs(d - 1.0 / np.pi) < 0.01

    def test_identity_b_edge(self):
        """density(±2, I) ≈ 0 (edge of support)."""
        b = np.eye(2)
        d = density_scalar_quadrature(2.0, b, eps=1e-2)
        assert d < 0.05  # near zero at the edge

    def test_outside_support(self):
        """density(3, I) ≈ 0 (outside support)."""
        b = np.eye(2)
        d = density_scalar_quadrature(3.0, b, eps=1e-2)
        assert d < 0.02

    def test_nonnegative(self):
        """Density should be nonnegative."""
        b = np.array([[1.0, 0.5], [0.5, 0.8]])
        for x in [-2.0, -1.0, 0.0, 0.5, 1.5, 3.0]:
            d = density_scalar_quadrature(x, b, eps=1e-2)
            assert d >= -1e-10, f"Negative density {d} at x={x}"


# ── diagonal b tests ────────────────────────────────────────────────────

class TestDiagonalB:
    """When b is diagonal, G_b(w) should also be diagonal for w = z*I."""

    def test_diagonal_b_gives_diagonal_G(self):
        b = np.diag([1.0, 0.5, 2.0])
        n = 3
        w = (0.5 + 0.1j) * np.eye(n, dtype=complex)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        # off-diagonal should be near zero
        off_diag_norm = la.norm(G - np.diag(np.diag(G)))
        assert off_diag_norm < 1e-6


# ── cross-check with fixed-point solver ─────────────────────────────────

class TestCrossCheckWithSolver:
    """
    Cross-validate the quadrature approach against the fixed-point solver
    for the unbiased semicircle (single Kraus operator b).
    """

    def test_scalar_density_matches_solver(self):
        """
        For a single Kraus operator A1 = b, the fixed-point solver's
        density should approximately match the quadrature density.
        """
        from free_matrix_laws import matrix_semicircle_density

        b = np.array([[1.0, 0.2], [0.2, 0.8]])
        A = [b]  # single Kraus operator
        x_test = 0.5
        eps = 1e-2

        d_solver = matrix_semicircle_density(x_test, A, eps=eps, tol=1e-11, maxiter=5000)
        d_quad = density_scalar_quadrature(x_test, b, eps=eps)

        # These are two independent methods; agreement to ~1e-2 is good
        assert abs(d_solver - d_quad) < 0.02, (
            f"Mismatch: solver={d_solver:.6f}, quadrature={d_quad:.6f}"
        )


# ── zero-matrix b test ──────────────────────────────────────────────────

class TestZeroB:
    """When b = 0, the resolvent is (w - 0)^{-1} = w^{-1} independent of x,
    and the integral reduces to w^{-1} (the measure has total mass 1)."""

    def test_zero_b(self):
        n = 2
        b = np.zeros((n, n))
        w = (0.5 + 0.1j) * np.eye(n, dtype=complex)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3)
        expected = la.inv(w)
        assert la.norm(G - expected) < 1e-3


# ── quad_opts forwarding ────────────────────────────────────────────────

class TestQuadOpts:
    def test_quad_opts_accepted(self):
        """Verify quad_opts is forwarded without error."""
        b = np.eye(2)
        w = (0.5 + 0.1j) * np.eye(2, dtype=complex)
        G = cauchy_matrix_semicircle_bruteforce(w, b, eps=1e-3, quad_opts={"limit": 50})
        assert G.shape == (2, 2)