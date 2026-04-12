"""
Tests for free_matrix_laws.singularity

Run with:  python -m pytest tests/test_singularity.py -v
"""

import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws.singularity import (
    solve_hms,
    hms_sweep,
    puiseux_exponent,
    singularity_report,
)
from free_matrix_laws.opvalued import covariance_map as eta


# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def cusp_pencil():
    """A1 = swap, A2 = diag(0,1) — singular with exponent -1/3."""
    A1 = np.array([[0, 1], [1, 0]], dtype=float)
    A2 = np.array([[0, 0], [0, 1]], dtype=float)
    return [A1, A2]


@pytest.fixture
def cusp_pencil_congruent():
    """Congruent to cusp_pencil via swap: A1 = swap, A2 = diag(1,0)."""
    A1 = np.array([[0, 1], [1, 0]], dtype=float)
    A2 = np.array([[1, 0], [0, 0]], dtype=float)
    return [A1, A2]


@pytest.fixture
def standard_semicircle_2x2():
    """A = I (doubly stochastic, non-singular, f(0) = 1/pi)."""
    return [np.eye(2)]


@pytest.fixture
def standard_semicircle_3x3():
    """A = I_3."""
    return [np.eye(3)]


@pytest.fixture
def diagonal_pencil():
    """A1 = diag(1,0), A2 = diag(0,1) — doubly stochastic, non-singular."""
    A1 = np.diag([1.0, 0.0])
    A2 = np.diag([0.0, 1.0])
    return [A1, A2]


# ── solve_hms ──────────────────────────────────────────────────────────

class TestSolveHMS:

    def test_residual_small(self, cusp_pencil):
        """Newton solver achieves small residual at moderate u."""
        W, iters, res = solve_hms(0.1, cusp_pencil)
        assert res < 1e-12

    def test_equation_satisfied(self, cusp_pencil):
        """Solution actually satisfies eta(W)W + uW = I."""
        u = 0.05
        W, _, _ = solve_hms(u, cusp_pencil, tol=1e-14)
        n = W.shape[0]
        lhs = eta(W, cusp_pencil) @ W + u * W
        np.testing.assert_allclose(lhs, np.eye(n), atol=1e-12)

    def test_accretive(self, cusp_pencil):
        """Solution is strictly accretive (Re W > 0)."""
        W, _, _ = solve_hms(0.1, cusp_pencil)
        ReW = 0.5 * (W + W.conj().T)
        eigs = la.eigvalsh(ReW)
        assert np.all(eigs > 0)

    def test_matches_exact_cubic(self, cusp_pencil):
        """For the cusp pencil, W is diagonal and d satisfies a cubic."""
        for u in [0.1, 0.01, 0.001]:
            W, _, _ = solve_hms(u, cusp_pencil, tol=1e-14)
            d = W[1, 1].real
            # d^3 + 2u d^2 + u^2 d - u = 0
            residual = d**3 + 2*u*d**2 + u**2*d - u
            assert abs(residual) < 1e-10, \
                f"Cubic residual {residual} at u={u}"

    def test_standard_semicircle_identity(self, standard_semicircle_2x2):
        """For eta = id (A = I), W(u) = w(u) I with w = (-u + sqrt(u^2+4))/2."""
        u = 0.3
        W, _, _ = solve_hms(u, standard_semicircle_2x2, tol=1e-14)
        w_exact = (-u + np.sqrt(u**2 + 4)) / 2
        np.testing.assert_allclose(W, w_exact * np.eye(2), atol=1e-10)

    def test_warm_start_helps(self, cusp_pencil):
        """Warm-starting from a nearby solution converges faster."""
        W1, it1, _ = solve_hms(0.01, cusp_pencil)
        W2_cold, it2_cold, _ = solve_hms(0.009, cusp_pencil)
        W2_warm, it2_warm, _ = solve_hms(0.009, cusp_pencil, W0=W1)
        assert it2_warm <= it2_cold

    def test_3x3(self, standard_semicircle_3x3):
        """Solver works for 3x3."""
        W, _, res = solve_hms(0.1, standard_semicircle_3x3)
        assert res < 1e-12
        assert W.shape == (3, 3)


# ── hms_sweep ──────────────────────────────────────────────────────────

class TestHMSSweep:

    def test_output_shape(self, cusp_pencil):
        """Sweep returns arrays of correct shape."""
        result = hms_sweep(cusp_pencil, n_points=30, u_min=1e-3)
        assert result["u"].shape == (30,)
        assert result["tr_W"].shape == (30,)
        assert result["eigs"].shape == (30, 2)
        assert result["det_W"].shape == (30,)
        assert result["residuals"].shape == (30,)
        assert result["converged"].shape == (30,)

    def test_u_descending(self, standard_semicircle_2x2):
        """u values are in descending order."""
        result = hms_sweep(standard_semicircle_2x2, n_points=20)
        assert np.all(np.diff(result["u"]) < 0)

    def test_all_converge_nonsingular(self, standard_semicircle_2x2):
        """Non-singular case: all points should converge."""
        result = hms_sweep(standard_semicircle_2x2, n_points=50,
                           u_min=1e-6)
        assert np.all(result["converged"])

    def test_tr_W_monotone_singular(self, cusp_pencil):
        """For singular case, tr W should increase as u decreases."""
        result = hms_sweep(cusp_pencil, n_points=50, u_min=1e-4)
        conv = result["converged"]
        tr = result["tr_W"][conv]
        # tr should be increasing (u is decreasing)
        assert np.all(np.diff(tr) > 0)


# ── puiseux_exponent ───────────────────────────────────────────────────

class TestPuiseuxExponent:

    def test_exact_power_law(self):
        """Recovers exact exponent from synthetic data u^alpha."""
        u = np.logspace(-1, -6, 100)
        for alpha in [-1/3, -1/2, 1/4, 0.0]:
            f = 2.0 * u**alpha if alpha != 0 else 2.0 * np.ones_like(u)
            est, C, R2 = puiseux_exponent(u, f)
            assert abs(est - alpha) < 0.01, \
                f"alpha={alpha}: estimated {est}"
            assert R2 > 0.99 or abs(alpha) < 0.01

    def test_with_convergence_mask(self):
        """Respects convergence mask."""
        u = np.logspace(-1, -6, 100)
        f = u**(-0.5)
        conv = np.ones(100, dtype=bool)
        conv[-10:] = False  # last 10 "did not converge"
        est, _, _ = puiseux_exponent(u, f, converged=conv)
        assert abs(est - (-0.5)) < 0.02

    def test_noisy_data(self):
        """Robust to mild noise."""
        rng = np.random.default_rng(42)
        u = np.logspace(-1, -5, 200)
        f = 3.0 * u**(-1/3) * (1 + 0.001 * rng.standard_normal(200))
        est, _, R2 = puiseux_exponent(u, f)
        assert abs(est - (-1/3)) < 0.02
        assert R2 > 0.999


# ── singularity_report ─────────────────────────────────────────────────

class TestSingularityReport:

    def test_cusp_singular(self, cusp_pencil):
        """Cusp pencil is detected as singular with exponent -1/3."""
        r = singularity_report(cusp_pencil, n_points=200,
                               u_min=1e-6, verbose=False)
        assert r["singular"]
        assert abs(r["alpha_tr"] - (-1/3)) < 0.01
        assert r["R2_tr"] > 0.999

    def test_congruent_same_exponent(self, cusp_pencil,
                                     cusp_pencil_congruent):
        """Congruent pencils have the same tr W exponent."""
        r1 = singularity_report(cusp_pencil, n_points=150,
                                u_min=1e-5, verbose=False)
        r2 = singularity_report(cusp_pencil_congruent, n_points=150,
                                u_min=1e-5, verbose=False)
        assert abs(r1["alpha_tr"] - r2["alpha_tr"]) < 0.01

    def test_standard_nonsingular(self, standard_semicircle_2x2):
        """Standard semicircle is non-singular with f(0) = 1/pi."""
        r = singularity_report(standard_semicircle_2x2, n_points=100,
                               u_min=1e-6, verbose=False)
        assert not r["singular"]
        assert abs(r["C_tr"] - 1.0) < 0.01  # tr W(0) = 1
        # f(0) = tr W(0) / pi = 1/pi
        assert abs(r["C_tr"] / np.pi - 1/np.pi) < 0.01

    def test_diagonal_nonsingular(self, diagonal_pencil):
        """Diagonal pencil diag(1,0) + diag(0,1) is non-singular."""
        r = singularity_report(diagonal_pencil, n_points=100,
                               u_min=1e-6, verbose=False)
        assert not r["singular"]

    def test_eigenvalue_exponents_sum(self, cusp_pencil):
        """Sum of eigenvalue exponents ≈ det exponent (for 2x2)."""
        r = singularity_report(cusp_pencil, n_points=200,
                               u_min=1e-6, verbose=False)
        # det W = eig1 * eig2, so alpha_det ≈ alpha_eig1 + alpha_eig2
        sum_eig = sum(r["alpha_eigs"])
        assert abs(sum_eig - r["alpha_det"]) < 0.05

    def test_3x3_nonsingular(self, standard_semicircle_3x3):
        """Works for 3x3 matrices."""
        r = singularity_report(standard_semicircle_3x3, n_points=80,
                               u_min=1e-5, verbose=False)
        assert not r["singular"]
        assert len(r["alpha_eigs"]) == 3

    def test_verbose_runs(self, cusp_pencil, capsys):
        """verbose=True prints output without error."""
        singularity_report(cusp_pencil, n_points=50,
                           u_min=1e-3, verbose=True)
        captured = capsys.readouterr()
        assert "Singularity report" in captured.out
        assert "SINGULAR" in captured.out