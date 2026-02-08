import numpy as np
import pytest

from free_matrix_laws.transforms import semicircle_cauchy_scalar


@pytest.mark.parametrize(
    "z",
    [
        0.3 + 0.7j,
        -1.2 + 0.1j,
        3.0 + 0.4j,
        -4.5 + 2.0j,
        0.3 - 0.7j,
        -1.2 - 0.1j,
        3.0 - 0.4j,
        -4.5 - 2.0j,
    ],
)
def test_herglotz_sign(z):
    """
    Herglotz property for the semicircle Stieltjes transform:
        Im(z) > 0  =>  Im(G(z)) < 0
        Im(z) < 0  =>  Im(G(z)) > 0
    """
    g = complex(semicircle_cauchy_scalar(z))
    assert g.imag * z.imag < 0


@pytest.mark.parametrize(
    "z",
    [
        0.2 + 1.0j,
        -0.8 + 0.3j,
        10.0 + 0.5j,
        -5.0 + 2.0j,
        0.2 - 1.0j,
        -0.8 - 0.3j,
    ],
)
def test_quadratic_identity(z):
    """
    For the standard semicircle (support [-2,2]):
        G(z)^2 - z G(z) + 1 = 0.
    """
    g = complex(semicircle_cauchy_scalar(z))
    resid = g * g - z * g + 1.0
    assert abs(resid) < 1e-12


@pytest.mark.parametrize("z", [1e2 + 3.0j, 1e3 + 2.0j, -1e3 + 5.0j])
def test_asymptotic_1_over_z(z):
    """
    As |z| -> infinity, G(z) = 1/z + O(1/z^3).
    """
    g = complex(semicircle_cauchy_scalar(z))
    target = 1.0 / z
    # Expected scale: |g - 1/z| ~ 1/|z|^3
    assert abs(g - target) < 5.0 / (abs(z) ** 3)


@pytest.mark.parametrize("x", [-1.7, -0.5, 0.0, 1.3])
def test_density_recovery_inside_support(x):
    """
    For x in (-2,2), with z = x + i eps (eps>0):
        rho(x) = -(1/pi) Im G(z)  as eps -> 0+
    Exact:
        rho(x) = (1/(2*pi)) * sqrt(4 - x^2).
    """
    eps = 1e-6
    z = x + 1j * eps
    g = complex(semicircle_cauchy_scalar(z))
    rho_est = -(g.imag) / np.pi
    rho_exact = 0.5 / np.pi * np.sqrt(max(0.0, 4.0 - x * x))
    assert abs(rho_est - rho_exact) < 5e-5


@pytest.mark.parametrize("x", [2.5, 3.0, -3.0])
def test_real_outside_support(x):
    """For real x with |x|>2, G(x) is real."""
    g = complex(semicircle_cauchy_scalar(x + 0j))
    assert abs(g.imag) < 1e-12


def test_conjugation_symmetry():
    """For a real measure: G(conj z) = conj G(z)."""
    z = 0.4 + 0.9j
    g1 = complex(semicircle_cauchy_scalar(np.conjugate(z)))
    g2 = np.conjugate(complex(semicircle_cauchy_scalar(z)))
    assert abs(g1 - g2) < 1e-12


def test_vectorization_and_shape():
    """Accept numpy arrays and preserve shape."""
    z = np.array([0.2 + 1.0j, -0.3 + 0.4j, 3.0 + 0.1j], dtype=np.complex128)
    out = semicircle_cauchy_scalar(z)
    assert isinstance(out, np.ndarray)
    assert out.shape == z.shape
    assert np.iscomplexobj(out)
