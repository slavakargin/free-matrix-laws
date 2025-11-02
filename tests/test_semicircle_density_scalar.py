import numpy as np
import numpy.testing as npt

from free_matrix_laws import semicircle_density_scalar, semicircle_density

def test_closed_form_values_c1():
    # c=1: f(0)=1/pi; f(±2)=0
    c = 1.0
    npt.assert_allclose(semicircle_density_scalar(0.0, c), 1/np.pi, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(semicircle_density_scalar(2.0, c), 0.0, rtol=0, atol=0)
    npt.assert_allclose(semicircle_density_scalar(-2.0, c), 0.0, rtol=0, atol=0)

def test_vectorized_input_and_shape():
    c = 2.5
    xs = np.array([-10.0, 0.0, 10.0])
    out = semicircle_density_scalar(xs, c)
    assert isinstance(out, np.ndarray)
    assert out.shape == xs.shape
    assert np.all(out >= 0)

def test_normalization_by_trapz():
    # ∫ f(x) dx = 1 over support [-2√c, 2√c]
    c = 1.7
    R = 2*np.sqrt(c)
    xs = np.linspace(-R, R, 2001)
    fx = semicircle_density_scalar(xs, c)
    Z = np.trapz(fx, xs)
    npt.assert_allclose(Z, 1.0, rtol=5e-3, atol=5e-3)

def test_invalid_c_raises():
    for bad in (0.0, -1.0):
        try:
            semicircle_density_scalar(0.0, bad)
            assert False, "expected ValueError for c<=0"
        except ValueError:
            pass

def test_reduction_to_matrix_case_sigmaI():
    # When A_i = sigma * I, the matrix density reduces to scalar with c=sigma^2
    n, sigma = 6, 1.3
    A = [sigma * np.eye(n)]
    c = sigma**2
    for x in (-1.0, 0.0, 1.0):
        f_mat = semicircle_density(x, A, eps=1e-2, tol=1e-12, maxiter=500)
        f_sca = semicircle_density_scalar(x, c=c)
        npt.assert_allclose(f_mat, f_sca, rtol=2e-2, atol=2e-3)
