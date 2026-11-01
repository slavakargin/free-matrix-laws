"""Tests for plot_brown_density and its automatic window."""
import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from free_matrix_laws import plot_brown_density, brown_density_grid  # noqa: E402
from free_matrix_laws.brown import _numerical_range_box  # noqa: E402

S2 = np.sqrt(2)
CIRCULAR = (np.zeros((1, 1)), [np.eye(1) / S2, 1j * np.eye(1) / S2])
E11, E22 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
COUPLED = (np.array([[0.0, 1.0], [0.3, 0.0]]),
           [0.42 * E11, 0.42j * E11, 0.42 * E22, 0.42j * E22])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestNumericalRangeBox:
    def test_circular(self):
        # Re c = s/sqrt2 and Im c = s'/sqrt2 have spectrum [-sqrt2, sqrt2]
        box = _numerical_range_box(*CIRCULAR)
        assert np.allclose(box, [-S2, S2, -S2, S2])

    def test_contains_random_matrix_spectrum(self):
        from free_matrix_laws import random_semicircle
        a0, a = COUPLED
        x0, x1, y0, y1 = _numerical_range_box(a0, a)
        N, rng = 300, np.random.default_rng(1)
        S = np.kron(a0, np.eye(N)) + sum(
            np.kron(ai, random_semicircle(N, field="complex", seed=rng)) for ai in a)
        ev = np.linalg.eigvals(S)
        tol = 0.15                       # finite-N fluctuations of the edge
        assert ev.real.min() > x0 - tol and ev.real.max() < x1 + tol
        assert ev.imag.min() > y0 - tol and ev.imag.max() < y1 + tol


class TestPlot:
    def test_returns_axes_with_heatmap(self):
        ax = plot_brown_density(*CIRCULAR, n=20)
        assert len(ax.collections) == 1                      # the QuadMesh
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        # the window contains the unit disk and is cropped below the sqrt2 box
        assert x0 < -1 and x1 > 1 and y0 < -1 and y1 > 1
        assert x1 - x0 < 2 * S2 + 0.2

    def test_auto_window_crops_non_normal_example(self):
        a0, a = COUPLED
        ax = plot_brown_density(a0, a, n=20)
        box = _numerical_range_box(a0, a)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        assert y1 - y0 < 0.8 * (box[3] - box[2])          # cropped in y
        assert x0 < -0.98 and x1 > 0.98 and y0 < -0.45 and y1 > 0.45   # support kept

    def test_explicit_limits_and_values(self):
        a0, a = CIRCULAR
        ax = plot_brown_density(a0, a, xlim=(-1.5, 1.5), ylim=(-1.0, 1.0), n=16, colorbar=False)
        assert np.allclose(ax.get_xlim(), (-1.5, 1.5)) and np.allclose(ax.get_ylim(), (-1.0, 1.0))
        mesh = ax.collections[0]
        xs, ys = np.linspace(-1.5, 1.5, 16), np.linspace(-1.0, 1.0, 11)
        expected = brown_density_grid(xs, ys, a0, a, eps=1e-3)
        assert np.allclose(np.asarray(mesh.get_array()).reshape(expected.shape), expected)

    def test_boundary_overlay(self):
        ax = plot_brown_density(*CIRCULAR, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n=40, boundary=True)
        assert len(ax.lines) == 1
        pts = ax.lines[0].get_xdata() + 1j * ax.lines[0].get_ydata()
        assert len(pts) > 8 and np.abs(np.abs(pts) - 1).max() < 1e-4

    def test_existing_axes(self):
        fig, ax = plt.subplots()
        assert plot_brown_density(*CIRCULAR, n=12, ax=ax, colorbar=False) is ax

    def test_self_adjoint_plots_spectral_density(self):
        a0 = np.array([[0.3, 0.1], [0.1, -0.2]])
        A = [np.array([[1.0, 0.2], [0.2, 0.5]]), np.array([[0.0, 0.4], [0.4, -0.3]])]
        ax = plot_brown_density(a0, A, n=40)
        assert len(ax.lines) == 1
        x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        assert len(x) == 200
        assert abs(np.trapezoid(y, x) - 1) < 5e-3
