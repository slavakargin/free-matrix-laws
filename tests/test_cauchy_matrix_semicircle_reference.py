# tests/test_cauchy_matrix_semicircle_reference.py

import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws.transforms import (
    cauchy_matrix_semicircle_reference,
    semicircle_cauchy_scalar,
)


def _omega():
    # A safe spectral parameter: Im(omega) > 0
    return 0.7 + 0.25j


def test_b_zero_gives_inverse():
    n = 3
    w = _omega() * np.eye(n, dtype=np.complex128)
    b = np.zeros((n, n), dtype=np.complex128)

    G = cauchy_matrix_semicircle_reference(w, b, n_quad=256)
    assert np.allclose(G, la.inv(w), rtol=1e-12, atol=1e-12)


def test_scalar_case_matches_scaled_scalar_cauchy():
    # n=1: E[(w - beta X)^(-1)] = (1/beta) * G_sc(w/beta)
    omega = _omega()
    beta = 2.0
    w = np.array([[omega]], dtype=np.complex128)
    b = np.array([[beta]], dtype=np.complex128)

    G = cauchy_matrix_semicircle_reference(w, b, n_quad=512)[0, 0]
    expected = (1.0 / beta) * semicircle_cauchy_scalar(omega / beta)

    assert abs(G - expected) < 5e-10


def test_b_identity_reduces_to_scalar_times_identity():
    n = 4
    omega = _omega()
    w = omega * np.eye(n, dtype=np.complex128)
    b = np.eye(n, dtype=np.complex128)

    G = cauchy_matrix_semicircle_reference(w, b, n_quad=512)
    expected = semicircle_cauchy_scalar(omega) * np.eye(n, dtype=np.complex128)

    assert np.allclose(G, expected, rtol=1e-10, atol=1e-10)


def test_diagonal_b_matches_entrywise_formula():
    omega = _omega()
    lam = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    n = lam.size

    w = omega * np.eye(n, dtype=np.complex128)
    b = np.diag(lam).astype(np.complex128)

    G = cauchy_matrix_semicircle_reference(w, b, n_quad=768)

    diag_expected = np.array(
        [(1.0 / l) * semicircle_cauchy_scalar(omega / l) for l in lam],
        dtype=np.complex128,
    )
    expected = np.diag(diag_expected)

    assert np.allclose(G, expected, rtol=2e-10, atol=2e-10)


def test_conjugation_symmetry_for_real_b_when_w_scalar_times_I():
    # For a real measure: G(conj omega) = G(omega)^*
    omega = _omega()
    n = 3
    w = omega * np.eye(n, dtype=np.complex128)

    b = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 2.0],
         [0.0, 2.0, -1.0]],
        dtype=np.complex128,
    )

    G = cauchy_matrix_semicircle_reference(w, b, n_quad=512)
    w_conj = np.conjugate(omega) * np.eye(n, dtype=np.complex128)
    G_conj = cauchy_matrix_semicircle_reference(w_conj, b, n_quad=512)

    assert np.allclose(G_conj, G.conj().T, rtol=5e-10, atol=5e-10)


def test_convergence_in_n_quad_sanity_1():
    # Sanity check: increasing n_quad should stabilize the result
    omega = _omega()
    n = 3
    w = omega * np.eye(n, dtype=np.complex128)
    b = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 2.0]],
        dtype=np.complex128,
    )

    G1 = cauchy_matrix_semicircle_reference(w, b, n_quad=128)
    G2 = cauchy_matrix_semicircle_reference(w, b, n_quad=1024)

    denom = max(1e-14, la.norm(G2, ord="fro"))
    rel = la.norm(G2 - G1, ord="fro") / denom
    assert rel < 2e-7

def test_convergence_in_n_quad_sanity_2():
    # More robust: compare against a high-n_quad reference and require improvement.
    omega = 0.7 + 0.5j  # slightly larger Im part => better conditioning
    n = 3
    w = omega * np.eye(n, dtype=np.complex128)
    b = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 2.0]],
        dtype=np.complex128,
    )

    G_lo = cauchy_matrix_semicircle_reference(w, b, n_quad=128)
    G_md = cauchy_matrix_semicircle_reference(w, b, n_quad=512)
    G_hi = cauchy_matrix_semicircle_reference(w, b, n_quad=2048)  # reference

    denom = max(1e-14, la.norm(G_hi, ord="fro"))
    err_lo = la.norm(G_lo - G_hi, ord="fro") / denom
    err_md = la.norm(G_md - G_hi, ord="fro") / denom

    # Require that increasing n_quad actually improves the approximation
    assert err_md < 0.7 * err_lo
    # And that the mid-level accuracy is within a reasonable tolerance
    assert err_md < 2e-7
