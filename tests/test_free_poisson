"""Tests for the free Poisson (Marchenko--Pastur) scalar helpers and their use
in the Kronecker / subordination machinery."""
import numpy as np
import numpy.linalg as la
import numpy.testing as npt
import pytest
from functools import partial
from scipy.integrate import quad, trapezoid

from free_matrix_laws import (
    free_poisson_cauchy_scalar,
    free_poisson_density_scalar,
    cauchy_kronecker,
    h_kronecker,
    subordination_kronecker,
    lambda_eps,
)


# ════════════════════════════════════════════════════════════════════════
# Scalar Cauchy transform
# ════════════════════════════════════════════════════════════════════════

class TestFreePoissonCauchyScalar:
    @pytest.mark.parametrize(
        "z",
        [0.3 + 0.7j, -1.2 + 0.1j, 3.0 + 0.4j, 5.0 + 2.0j,
         0.3 - 0.7j, -1.2 - 0.1j, 3.0 - 0.4j, 5.0 - 2.0j],
    )
    @pytest.mark.parametrize("lam", [0.5, 1.0, 4.0])
    def test_herglotz_sign(self, z, lam):
        """Im(z) > 0 => Im(G(z)) < 0, and vice versa."""
        g = complex(free_poisson_cauchy_scalar(z, lam))
        assert g.imag * z.imag < 0

    @pytest.mark.parametrize(
        "z",
        [0.2 + 1.0j, -0.8 + 0.3j, 3.0 + 0.5j, 7.0 + 0.2j, 0.4 - 0.9j],
    )
    @pytest.mark.parametrize("lam", [1.0, 2.5, 4.0])
    def test_marchenko_pastur_quadratic(self, z, lam):
        r"""For lam >= 1 (no atom) the transform satisfies
        z G^2 - (1 + z - lam) G + 1 = 0."""
        g = complex(free_poisson_cauchy_scalar(z, lam))
        resid = z * g * g - (1.0 + z - lam) * g + 1.0
        assert abs(resid) < 1e-11

    @pytest.mark.parametrize("z", [1e2 + 3.0j, 1e3 + 2.0j, -1e3 + 5.0j])
    @pytest.mark.parametrize("lam", [0.5, 1.0, 4.0])
    def test_asymptotic_first_moment(self, z, lam):
        """G(z) = 1/z + lam/z^2 + ...  =>  z*(z*G - 1) -> lam (mean)."""
        g = complex(free_poisson_cauchy_scalar(z, lam))
        moment_est = z * (z * g - 1.0)
        assert abs(moment_est - lam) < 50.0 / abs(z)

    @pytest.mark.parametrize("lam", [0.5, 4.0])
    def test_density_recovery_inside_support(self, lam):
        """-(1/pi) Im G(x + i eps) -> free Poisson density on the bulk."""
        a = (1 - np.sqrt(lam))**2
        b = (1 + np.sqrt(lam))**2
        xs = np.linspace(a + 0.1 * (b - a), b - 0.1 * (b - a), 7)
        eps = 1e-6
        rho_est = -free_poisson_cauchy_scalar(xs + 1j * eps, lam).imag / np.pi
        rho_exact = free_poisson_density_scalar(xs, lam)
        npt.assert_allclose(rho_est, rho_exact, rtol=1e-3, atol=1e-4)

    def test_atom_mass_for_small_lambda(self):
        """For lam < 1 there is an atom of mass (1 - lam) at the origin:
        lim_{z->0} z G(z) = 1 - lam."""
        lam = 0.4
        z = 1e-8j  # approach the origin from the upper half-plane
        zg = complex(z * free_poisson_cauchy_scalar(z, lam))
        npt.assert_allclose(zg.real, 1.0 - lam, atol=1e-4)

    def test_conjugation_symmetry(self):
        """Real measure => G(conj z) = conj G(z)."""
        z = 0.4 + 0.9j
        g1 = complex(free_poisson_cauchy_scalar(np.conjugate(z), 4.0))
        g2 = np.conjugate(complex(free_poisson_cauchy_scalar(z, 4.0)))
        assert abs(g1 - g2) < 1e-12

    def test_vectorization_and_shape(self):
        z = np.array([0.2 + 1.0j, -0.3 + 0.4j, 3.0 + 0.1j], dtype=np.complex128)
        out = free_poisson_cauchy_scalar(z, 4.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == z.shape
        assert np.iscomplexobj(out)

    def test_invalid_lambda_raises(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                free_poisson_cauchy_scalar(0.5 + 0.1j, bad)


# ════════════════════════════════════════════════════════════════════════
# Scalar density
# ════════════════════════════════════════════════════════════════════════

class TestFreePoissonDensityScalar:
    @pytest.mark.parametrize("lam", [0.5, 1.0, 4.0])
    def test_edges_are_zero(self, lam):
        a = (1 - np.sqrt(lam))**2
        b = (1 + np.sqrt(lam))**2
        npt.assert_allclose(free_poisson_density_scalar(a, lam), 0.0, atol=1e-12)
        npt.assert_allclose(free_poisson_density_scalar(b, lam), 0.0, atol=1e-12)

    @pytest.mark.parametrize("lam,expected_mass", [(4.0, 1.0), (1.0, 1.0), (0.5, 0.5)])
    def test_absolutely_continuous_mass(self, lam, expected_mass):
        """a.c. mass is 1 for lam >= 1, and lam for lam < 1 (atom carries 1-lam).
        Uses adaptive quadrature to handle the sqrt edge singularities."""
        a = (1 - np.sqrt(lam))**2
        b = (1 + np.sqrt(lam))**2
        mass, _ = quad(lambda x: free_poisson_density_scalar(x, lam), a, b,
                       points=[a, b])
        npt.assert_allclose(mass, expected_mass, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("lam", [0.5, 4.0])
    def test_first_moment(self, lam):
        """int x f(x) dx = lam (the atom at 0 contributes nothing)."""
        a = (1 - np.sqrt(lam))**2
        b = (1 + np.sqrt(lam))**2
        m1, _ = quad(lambda x: x * free_poisson_density_scalar(x, lam), a, b,
                     points=[a, b])
        npt.assert_allclose(m1, lam, rtol=1e-4, atol=1e-4)

    def test_vectorized_nonnegative_shape(self):
        xs = np.array([-1.0, 0.5, 1.0, 9.5, 20.0])
        out = free_poisson_density_scalar(xs, 4.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == xs.shape
        assert np.all(out >= 0.0)

    def test_invalid_lambda_raises(self):
        for bad in (0.0, -2.0):
            with pytest.raises(ValueError):
                free_poisson_density_scalar(1.0, bad)


# ════════════════════════════════════════════════════════════════════════
# End-to-end: anticommutator of two free Poisson via subordination
# ════════════════════════════════════════════════════════════════════════

# Standard anticommutator linearization data
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
SUB_EPS = 1e-4


class TestFreePoissonAnticommutator:
    def test_kronecker_matches_regularization_consistency(self):
        """cauchy_kronecker with the Poisson scalar transform is self-consistent:
        smaller eps changes the answer only at O(eps)."""
        lam = 4.0
        Gp = partial(free_poisson_cauchy_scalar, lam=lam)
        w = (5.0 + 0.05j) * np.eye(3)
        G_coarse = cauchy_kronecker(w, A1, cauchy_scalar=Gp, eps=1e-4)
        G_fine = cauchy_kronecker(w, A1, cauchy_scalar=Gp, eps=1e-6)
        assert la.norm(G_fine - G_coarse) < 1e-3

    def test_subordination_fixed_point(self):
        """omega should satisfy omega = h_Y(h_X(omega) + b) + b for free Poisson."""
        lam = 4.0
        Gp = partial(free_poisson_cauchy_scalar, lam=lam)
        z = 5.0 + 0.1j
        b = lambda_eps(z, 3) - A0
        omega = subordination_kronecker(
            b, A1, A2, cauchy_scalar_x=Gp, cauchy_scalar_y=Gp, eps=SUB_EPS
        )
        W1 = h_kronecker(omega, A1, cauchy_scalar=Gp, eps=SUB_EPS) + b
        W2 = h_kronecker(W1, A2, cauchy_scalar=Gp, eps=SUB_EPS) + b
        assert np.allclose(omega, W2, atol=1e-6), (
            f"Fixed-point residual: {la.norm(omega - W2):.2e}"
        )

    def test_density_normalization_and_mean(self):
        """The anticommutator density of two free Poisson(lam) integrates to 1
        and has mean 2*lam^2 (since E[XY+YX] = 2 E[X] E[Y] = 2 lam^2)."""
        lam = 4.0
        Gp = partial(free_poisson_cauchy_scalar, lam=lam)

        def density(x, eps=0.01):
            z = x + eps * 1j
            b = lambda_eps(z, 3) - A0
            omega = subordination_kronecker(
                b, A1, A2, cauchy_scalar_x=Gp, cauchy_scalar_y=Gp
            )
            G = cauchy_kronecker(omega, A1, cauchy_scalar=Gp)
            return -G[0, 0].imag / np.pi

        xs = np.linspace(0.0, 110.0, 60)
        f = np.array([density(x) for x in xs])
        # eps=0.01 edge smoothing can yield tiny negative values near the tails
        assert f.min() > -5e-3

        mass = trapezoid(f, xs)
        mean = trapezoid(xs * f, xs)
        npt.assert_allclose(mass, 1.0, atol=5e-2)
        npt.assert_allclose(mean, 2.0 * lam**2, rtol=5e-2)