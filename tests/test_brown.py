"""Tests for free_matrix_laws.brown: regularized Brown density and
logarithmic potential of (biased) matrix semicircles via hermitization."""
import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws import (
    brown_density,
    brown_density_grid,
    brown_log_potential,
    cauchy_hermitized_matrix_semicircle,
    hermitize,
)
from free_matrix_laws.brown import _eta_matrix, _eta_apply


def circular():
    """S = c, standard circular element: a_1 = 1/sqrt2, a_2 = i/sqrt2."""
    return np.zeros((1, 1)), [np.eye(1) / np.sqrt(2), 1j * np.eye(1) / np.sqrt(2)]


def elliptic(theta):
    """S = s_1 + i theta s_2."""
    return np.zeros((1, 1)), [np.eye(1), 1j * theta * np.eye(1)]


RNG = np.random.default_rng(7)
M2_A0 = RNG.normal(size=(2, 2)) + 1j * RNG.normal(size=(2, 2))
M2_A = [(RNG.normal(size=(2, 2)) + 1j * RNG.normal(size=(2, 2))) / 2
        for _ in range(2)]


# ════════════════════════════════════════════════════════════════════════
# Hermitization and the eta matrix
# ════════════════════════════════════════════════════════════════════════

class TestHermitize:
    def test_shapes_and_hermiticity(self):
        A0, A = hermitize(0.3 + 0.2j, M2_A0, M2_A)
        assert A0.shape == (4, 4) and A.shape == (2, 4, 4)
        assert la.norm(A0 - A0.conj().T) < 1e-14
        for Ai in A:
            assert la.norm(Ai - Ai.conj().T) < 1e-14

    def test_eta_matrix_matches_apply(self):
        _, A = hermitize(0.1, M2_A0, M2_A)
        B = RNG.normal(size=(4, 4)) + 1j * RNG.normal(size=(4, 4))
        lhs = (_eta_matrix(A) @ B.ravel()).reshape(4, 4)
        assert la.norm(lhs - _eta_apply(B, A)) < 1e-12


# ════════════════════════════════════════════════════════════════════════
# Dyson solver
# ════════════════════════════════════════════════════════════════════════

class TestSolver:
    @pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
    def test_residual_and_block_structure(self, eps):
        z = 0.2 + 0.1j
        G, info = cauchy_hermitized_matrix_semicircle(
            z, M2_A0, M2_A, eps=eps, return_info=True)
        A0, A = hermitize(z, M2_A0, M2_A)
        res = la.inv(G) + _eta_apply(G, A) - (1j * eps * np.eye(4) - A0)
        assert la.norm(res) < 1e-11
        assert info["residual"] < 1e-11
        # -Im G > 0
        Y = (G - G.conj().T) / (2j)
        assert la.eigvalsh(Y).max() < 0
        # G_12 = G_21^*, tr G_11 = tr G_22, diagonal blocks purely imaginary-negative
        assert la.norm(G[:2, 2:] - G[2:, :2].conj().T) < 1e-11
        assert abs(np.trace(G[:2, :2]) - np.trace(G[2:, 2:])) < 1e-11
        for blk in (G[:2, :2], G[2:, 2:]):
            assert la.norm(blk + blk.conj().T) < 1e-11       # anti-Hermitian
            assert la.eigvalsh(1j * blk).min() > 0            # i*blk > 0

    def test_warm_start_fast(self):
        G0 = cauchy_hermitized_matrix_semicircle(0.3, M2_A0, M2_A, eps=1e-4)
        _, info1 = cauchy_hermitized_matrix_semicircle(
            0.31, M2_A0, M2_A, eps=1e-4, G0=G0, return_info=True)
        assert info1["iters"] <= 8
        assert info1["residual"] < 1e-11

    def test_inside_support_small_eps_m4(self):
        """Random 4x4 example inside the support: this is where the chiral
        soft mode makes plain Newton stall."""
        rng = np.random.default_rng(1)
        m = 4
        a0 = rng.normal(size=(m, m))
        a = [rng.normal(size=(m, m)) / np.sqrt(m) for _ in range(3)]
        _, info = cauchy_hermitized_matrix_semicircle(
            0.3, a0, a, eps=1e-4, return_info=True)
        assert info["residual"] < 1e-11
        assert info["iters"] < 40


# ════════════════════════════════════════════════════════════════════════
# Density: exact ground truth
# ════════════════════════════════════════════════════════════════════════

class TestCircular:
    @pytest.mark.parametrize("z", [0.0, 0.3 + 0.2j, -0.7j, 0.9])
    def test_inside(self, z):
        a0, a = circular()
        assert abs(brown_density(z, a0, a, eps=1e-5) - 1 / np.pi) < 1e-4

    @pytest.mark.parametrize("z", [1.1, 1.5 + 0.5j, -2.0])
    def test_outside(self, z):
        a0, a = circular()
        assert abs(brown_density(z, a0, a, eps=1e-5)) < 1e-6

    def test_log_potential(self):
        a0, a = circular()
        # log-potential of the uniform disk: |z|^2/2 - 1/2 inside, log|z| outside
        # inside the support the regularization bias is O(eps)
        assert abs(brown_log_potential(0.5, a0, a, eps=1e-5) - (0.125 - 0.5)) < 1e-4
        assert abs(brown_log_potential(2.0, a0, a, eps=1e-5) - np.log(2.0)) < 1e-8


class TestElliptic:
    @pytest.mark.parametrize("theta", [0.3, 0.6, 1.0])
    def test_uniform_density_inside(self, theta):
        # support: ellipse with semi-axes 2/sqrt(1+t^2), 2t^2/sqrt(1+t^2)
        # => area 4 pi t^2/(1+t^2), uniform density (1+t^2)/(4 pi t^2)
        a0, a = elliptic(theta)
        rho_exact = (1 + theta**2) / (4 * np.pi * theta**2)
        for z in [0.0, 0.2 + 0.05j * theta]:
            assert abs(brown_density(z, a0, a, eps=1e-5) - rho_exact) < 1e-3 * rho_exact

    def test_outside(self):
        a0, a = elliptic(0.5)
        assert abs(brown_density(3.0, a0, a, eps=1e-5)) < 1e-6


class TestBlockDiagonalMean:
    """S = a0 (x) 1 + 1 (x) c with a0 = diag(0, 2): the two blocks commute
    with c, so mu_S = 1/2 (uniform disk at 0) + 1/2 (uniform disk at 2)."""

    def test_density(self):
        a0 = np.diag([0.0, 2.0])
        a = [np.eye(2) / np.sqrt(2), 1j * np.eye(2) / np.sqrt(2)]
        for z in [0.0, 0.5j, 2.5, 1.9 + 0.3j]:
            assert abs(brown_density(z, a0, a, eps=1e-5) - 1 / (2 * np.pi)) < 1e-4
        assert abs(brown_density(1.0 + 1.5j, a0, a, eps=1e-5)) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Density: structural identities on a non-trivial example
# ════════════════════════════════════════════════════════════════════════

class TestIdentities:
    @pytest.mark.parametrize("z", [0.2 + 0.1j, 1.0 - 0.7j, 3.0 + 2.0j])
    def test_laplacian_of_log_potential(self, z):
        """rho_eps = (1/2pi) Laplacian L_eps exactly, for every eps > 0."""
        eps, h = 0.1, 1e-3
        L = lambda w: brown_log_potential(w, M2_A0, M2_A, eps=eps)
        lap = (L(z + h) + L(z - h) + L(z + 1j * h) + L(z - 1j * h) - 4 * L(z)) / h**2
        assert abs(lap / (2 * np.pi) - brown_density(z, M2_A0, M2_A, eps=eps)) < 1e-6

    def test_density_is_real(self):
        _, info = brown_density(0.4 - 0.3j, M2_A0, M2_A, eps=1e-4, return_info=True)
        assert abs(info["imag_part"]) < 1e-10

    def test_total_mass_and_mean(self):
        """Integrate rho_eps over a box containing the support."""
        a0 = np.array([[0.5, 1.0], [0.0, -0.5]])
        a = [np.eye(2) / np.sqrt(2), 1j * np.eye(2) / np.sqrt(2),
             np.array([[0, 1], [0, 0]]) * 0.5]
        xs = np.linspace(-3.5, 3.5, 71)
        ys = np.linspace(-3.0, 3.0, 61)
        rho = brown_density_grid(xs, ys, a0, a, eps=1e-2)
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        mass = rho.sum() * dx * dy
        X, Y = np.meshgrid(xs, ys)
        mean = ((X + 1j * Y) * rho).sum() * dx * dy
        assert abs(mass - 1.0) < 2e-2
        assert abs(mean - np.trace(a0) / 2) < 2e-2
        assert rho.min() > -1e-8


class TestTriangular:
    """Block upper-triangular S: the Brown measure is the average of the
    Brown measures of the diagonal blocks, here the circular elements
    +-1/2 + c, whatever the strictly upper-triangular coupling is."""

    def test_two_disks(self):
        a0 = np.array([[0.5, 1.0], [0.0, -0.5]])
        a = [np.eye(2) / np.sqrt(2), 1j * np.eye(2) / np.sqrt(2),
             np.array([[0, 1], [0, 0]]) * 0.5]
        # at z = 0 the hermitized G(z; i0) does not exist (norm ~ eps^{-1/3}),
        # and the regularization bias of the density is O(eps^{2/3})
        assert abs(brown_density(0.0, a0, a, eps=1e-6) - 1 / np.pi) < 1e-4
        assert abs(brown_density(1.2 + 0.3j, a0, a, eps=1e-5) - 1 / (2 * np.pi)) < 1e-4
        assert abs(brown_density(-1.2 - 0.3j, a0, a, eps=1e-5) - 1 / (2 * np.pi)) < 1e-4
        assert abs(brown_density(0.0 + 1.3j, a0, a, eps=1e-5)) < 1e-6
