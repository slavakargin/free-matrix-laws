"""Tests for biased_matrix_semicircle_logdet (log Fuglede--Kadison determinant
of the shifted matrix semicircle), computed via the closed Dyson formula."""
import numpy as np
import numpy.linalg as la
import pytest
from scipy.integrate import trapezoid

from free_matrix_laws import (
    biased_matrix_semicircle_logdet as bms_logdet,
    biased_matrix_semicircle_density as bms_density,
    random_semicircle,
)


def _scalar_logdet_exact(a, z, sigma):
    r"""Closed form for m=1: S = a + sigma * s, s standard semicircular.
    Inside support (|a-z| <= 2 sigma) and outside, from the scalar test."""
    d = abs(a - z)
    if d <= 2 * sigma:
        return np.log(sigma) - 0.5 + (a - z) ** 2 / (4 * sigma ** 2)
    root = np.sqrt(d ** 2 - 4 * sigma ** 2)
    bstar = (d + root) / 2
    return np.log(bstar) + d * (d - root) / (4 * sigma ** 2) - 0.5


# ════════════════════════════════════════════════════════════════════════
# Scalar case: exact ground truth
# ════════════════════════════════════════════════════════════════════════

class TestScalar:
    @pytest.mark.parametrize(
        "sigma,a,z",
        [(1.0, 0.0, 3.0), (1.0, 0.0, -3.5), (2.0, 1.0, 6.0), (1.5, -0.5, 5.0)],
    )
    def test_outside_support(self, sigma, a, z):
        A = np.array([[[sigma]]])
        a0 = np.array([[a]])
        val = bms_logdet(z, a0, A, eps=1e-4)
        assert abs(val - _scalar_logdet_exact(a, z, sigma)) < 1e-5

    @pytest.mark.parametrize(
        "sigma,a,z",
        [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.5, -0.5, 0.3), (1.0, 0.0, 1.8)],
    )
    def test_inside_support(self, sigma, a, z):
        A = np.array([[[sigma]]])
        a0 = np.array([[a]])
        val = bms_logdet(z, a0, A, eps=1e-4)
        # inside the support the i*eps regularization gives an O(eps) bias
        assert abs(val - _scalar_logdet_exact(a, z, sigma)) < 2e-3


# ════════════════════════════════════════════════════════════════════════
# Matrix case
# ════════════════════════════════════════════════════════════════════════

# Fixed 2x2, r=2 example (Hermitian coefficients and mean)
A_MAT = np.array([[[1.0, 0.2], [0.2, 0.5]],
                  [[0.0, 0.4], [0.4, -0.3]]])
A0_MAT = np.array([[0.3, 0.1], [0.1, -0.2]])


def _spectral_bound():
    return la.norm(A0_MAT, 2) + 2 * sum(la.norm(Ai, 2) for Ai in A_MAT)


class TestMatrixDeterministic:
    def test_matches_potential_integral_outside(self):
        """Outside the support, log det = int log|t-z| rho(t) dt with a smooth
        integrand. Cross-check the closed formula against this integral, where
        rho is obtained from biased_matrix_semicircle_density."""
        R = _spectral_bound()
        z = R + 2.0
        t = np.linspace(-R - 1.5, R + 1.5, 4000)
        rho = np.array([bms_density(ti, A0_MAT, A_MAT, eps=1e-3) for ti in t])
        pot = trapezoid(rho * np.log(np.abs(t - z)), t)
        val = bms_logdet(z, A0_MAT, A_MAT)
        assert abs(val - pot) < 5e-3

    def test_density_normalization(self):
        """Sanity: the density used above integrates to ~1."""
        R = _spectral_bound()
        t = np.linspace(-R - 1.5, R + 1.5, 4000)
        rho = np.array([bms_density(ti, A0_MAT, A_MAT, eps=1e-3) for ti in t])
        assert abs(trapezoid(rho, t) - 1.0) < 1e-2


class TestMatrixMonteCarlo:
    @pytest.mark.parametrize("z", [5.6, 0.5])
    def test_matches_random_matrix(self, z):
        """Independent ground truth: mean of (1/(mN)) sum log|lambda - z| over
        finite random realizations of S = a0 (x) I + sum A_i (x) s_i."""
        N, T = 300, 6
        vals = []
        for k in range(T):
            S = np.kron(A0_MAT, np.eye(N)).astype(complex)
            for j, Ai in enumerate(A_MAT):
                s = random_semicircle(N, field="complex", seed=100 * k + j)
                S = S + np.kron(Ai.astype(complex), s)
            S = (S + S.conj().T) / 2
            vals.append(np.mean(np.log(np.abs(la.eigvalsh(S) - z))))
        mc = float(np.mean(vals))
        val = bms_logdet(z, A0_MAT, A_MAT)
        assert abs(val - mc) < 3e-2


class TestAliasAndSanity:
    def test_alias_identity(self):
        import free_matrix_laws as fml
        assert fml.bms_logdet is fml.biased_matrix_semicircle_logdet

    def test_returns_python_float(self):
        val = bms_logdet(5.0, A0_MAT, A_MAT)
        assert isinstance(val, float)

    def test_eps_insensitive_outside_support(self):
        z = 8.0
        v1 = bms_logdet(z, A0_MAT, A_MAT, eps=1e-3)
        v2 = bms_logdet(z, A0_MAT, A_MAT, eps=1e-5)
        assert abs(v1 - v2) < 1e-4
