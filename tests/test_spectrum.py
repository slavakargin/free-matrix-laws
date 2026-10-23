"""Tests for the size-m spectrum criterion in free_matrix_laws.brown:
resolvent_matrix_semicircle, spectral_radius_T, spectrum_indicator,
spectrum_indicator_grid and spectrum_boundary."""
import warnings

import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws import (
    resolvent_matrix_semicircle,
    spectral_radius_T,
    spectrum_indicator,
    spectrum_indicator_grid,
    spectrum_boundary,
    brown_density,
)
from free_matrix_laws.brown import _kappa_apply


def circular(var=1.0):
    t = np.sqrt(var)
    return np.zeros((1, 1)), [t * np.eye(1) / np.sqrt(2), 1j * t * np.eye(1) / np.sqrt(2)]


def elliptic(theta):
    return np.zeros((1, 1)), [np.eye(1), 1j * theta * np.eye(1)]


def mean_plus_circular(eigs, var):
    """S = a0 (x) 1 + 1 (x) c with a0 = diag(eigs) and c of variance var:
    the spectrum is the union of the disks of radius sqrt(var) around eigs."""
    m = len(eigs)
    t = np.sqrt(var)
    return np.diag(eigs).astype(complex), [t * np.eye(m) / np.sqrt(2), 1j * t * np.eye(m) / np.sqrt(2)]


# ════════════════════════════════════════════════════════════════════════
# Resolvent equation and the map T_g
# ════════════════════════════════════════════════════════════════════════

class TestResolvent:
    def test_circular_closed_form(self):
        a0, a = circular()
        for z in [2.0, 1.2 + 0.3j, -3j]:
            g, info = resolvent_matrix_semicircle(z, a0, a, return_info=True)
            assert info["converged"]
            assert abs(g[0, 0] - 1 / z) < 1e-12          # kappa = 0 for c

    def test_elliptic_satisfies_equation(self):
        a0, a = elliptic(0.5)
        z = 2.5 + 0.4j
        g, info = resolvent_matrix_semicircle(z, a0, a, return_info=True)
        assert info["converged"]
        # g^{-1} + (1 - theta^2) g = z
        assert abs(1 / g[0, 0] + 0.75 * g[0, 0] - z) < 1e-11

    def test_matrix_case_residual_and_warm_start(self):
        rng = np.random.default_rng(3)
        a0 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        a = [(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) / 3 for _ in range(2)]
        z = 6.0 + 2.0j
        g, info = resolvent_matrix_semicircle(z, a0, a, return_info=True)
        assert info["converged"]
        res = la.inv(g) + _kappa_apply(g, np.array(a)) - (z * np.eye(3) - a0)
        assert la.norm(res) < 1e-10
        # large-z asymptotics g ~ (z - a0)^{-1}
        zz = 400.0
        g2 = resolvent_matrix_semicircle(zz, a0, a)
        assert la.norm(g2 - la.inv(zz * np.eye(3) - a0)) < 1e-4
        # warm start converges in a couple of steps
        _, info2 = resolvent_matrix_semicircle(z + 0.05, a0, a, g0=g, return_info=True)
        assert info2["converged"] and info2["iters"] <= 5

    def test_spectral_radius_T_scalar(self):
        _, a = circular(var=2.0)
        for g in [0.3, 0.7 - 0.2j]:
            assert abs(spectral_radius_T(np.array([[g]]), a) - 2.0 * abs(g) ** 2) < 1e-13

    def test_spectral_radius_T_is_perron_eigenvalue(self):
        rng = np.random.default_rng(5)
        a = [rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for _ in range(2)]
        g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        r = spectral_radius_T(g, a)
        # power iteration on a PSD matrix converges to the Perron eigenvalue
        B = np.eye(3, dtype=complex)
        for _ in range(300):
            B = sum((g @ ai) @ B @ (g @ ai).conj().T for ai in a)
            B /= la.norm(B)
        TB = sum((g @ ai) @ B @ (g @ ai).conj().T for ai in a)
        assert abs(la.norm(TB) - r) < 1e-8 * max(1, r)


# ════════════════════════════════════════════════════════════════════════
# Indicator: exact ground truth
# ════════════════════════════════════════════════════════════════════════

class TestIndicator:
    @pytest.mark.parametrize("z", [2.0, 1.2 + 0.3j, 1.01, -0.7 + 0.8j])
    def test_circular_outside(self, z):
        a0, a = circular()
        assert abs(spectrum_indicator(z, a0, a) - abs(z) ** -2) < 1e-10

    @pytest.mark.parametrize("z", [0.99, 0.5, 0.0, 0.3j])
    def test_circular_inside(self, z):
        a0, a = circular()
        assert spectrum_indicator(z, a0, a) >= 1.0

    @pytest.mark.parametrize("theta", [0.3, 0.6, 1.0])
    def test_elliptic_boundary(self, theta):
        a0, a = elliptic(theta)
        s = np.sqrt(1 + theta ** 2)
        for phi in [0.0, 0.4, 1.3, 2.2, np.pi]:
            zb = (2 * np.cos(phi) + 2j * theta ** 2 * np.sin(phi)) / s
            assert spectrum_indicator(1.03 * zb, a0, a) < 1.0
            assert spectrum_indicator(0.97 * zb, a0, a) >= 1.0

    def test_union_of_disks(self):
        a0, a = mean_plus_circular([0.0, 2.0], 1.0)
        for z in [1.0 + 0.3j, 1.0 + 1.1j, 3.1, 2.9, -0.5 + 0.9j, -1.2]:
            r = spectrum_indicator(z, a0, a)
            pred = max(abs(z) ** -2, abs(z - 2) ** -2)
            if pred < 1:
                assert abs(r - pred) < 1e-10
            else:
                assert r >= 1.0

    def test_hole_in_the_spectrum(self):
        """Four disks of radius 1.5 around 2, 2i, -2, -2i form a ring; the
        origin lies in a bounded component of the resolvent set, where g is
        not the continuation from infinity."""
        a0, a = mean_plus_circular([2.0, 2.0j, -2.0, -2.0j], 2.25)
        for z in [0.0, 0.3 + 0.2j, -0.4j]:
            pred = max(2.25 / abs(z - l) ** 2 for l in np.diag(a0))
            assert pred < 1
            assert abs(spectrum_indicator(z, a0, a) - pred) < 1e-9
        assert spectrum_indicator(1.0, a0, a) >= 1.0

    def test_deep_inside_without_hermitized_solution(self):
        """At an atom-like point the hermitized solver may not converge;
        the indicator must still report 'inside'."""
        a0, a = mean_plus_circular([0.0, 2.0], 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert spectrum_indicator(2.0, a0, a) >= 1.0


# ════════════════════════════════════════════════════════════════════════
# Grid and boundary
# ════════════════════════════════════════════════════════════════════════

class TestBoundary:
    def test_circle(self):
        a0, a = circular()
        xs = np.linspace(-1.5, 1.5, 13)
        pts, r = spectrum_boundary(xs, xs, a0, a, return_grid=True)
        assert r.shape == (13, 13)
        assert len(pts) > 20
        assert np.abs(np.abs(pts) - 1.0).max() < 1e-5

    def test_ellipse(self):
        theta = 0.5
        a0, a = elliptic(theta)
        s = np.sqrt(1 + theta ** 2)
        A, B = 2 / s, 2 * theta ** 2 / s
        pts = spectrum_boundary(np.linspace(-2.2, 2.2, 23), np.linspace(-1.0, 1.0, 11), a0, a)
        assert len(pts) > 20
        assert np.abs((pts.real / A) ** 2 + (pts.imag / B) ** 2 - 1).max() < 1e-4

    def test_ring_with_hole(self):
        a0, a = mean_plus_circular([2.0, 2.0j, -2.0, -2.0j], 2.25)
        xs = np.linspace(-4, 4, 25)
        pts = spectrum_boundary(xs, xs, a0, a)
        d = np.min(np.abs(pts[:, None] - np.diag(a0)[None, :]), axis=1)
        assert np.abs(d - 1.5).max() < 1e-5
        assert np.sum(np.abs(pts) < 1.5) > 8          # inner boundary found

    def test_boundary_matches_density_support(self):
        """Coupled example S = [[c1, 1], [delta, c2]]: the Brown density is
        O(1) just inside the boundary found by the criterion and ~0 just
        outside it."""
        E11, E22 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
        t = 0.6 / np.sqrt(2)
        a0 = np.array([[0.0, 1.0], [0.3, 0.0]])
        a = [t * E11, 1j * t * E11, t * E22, 1j * t * E22]
        pts = spectrum_boundary(np.linspace(-1.6, 1.6, 17), np.linspace(-1.2, 1.2, 13), a0, a)
        assert len(pts) > 20
        rng = np.random.default_rng(0)
        for z in rng.choice(pts, 6, replace=False):
            n = z / abs(z)                      # outward direction (star-shaped about 0)
            assert brown_density(z + 0.08 * n, a0, a, eps=1e-4) < 2e-3
            assert brown_density(z - 0.08 * n, a0, a, eps=1e-4) > 0.05
