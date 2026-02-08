import numpy as np
import numpy.linalg as la
import pytest

from free_matrix_laws.transforms import (
    matrix_cauchy_semicircle_reference,
    semicircle_cauchy_scalar,
)


def _omega():
    # Хороший "безопасный" параметр: Im(omega) > 0
    return 0.7 + 0.25j


def test_b_zero_gives_inverse():
    n = 3
    w = _omega() * np.eye(n, dtype=np.complex128)
    b = np.zeros((n, n), dtype=np.complex128)

    G = matrix_cauchy_semicircle_reference(w, b, n_quad=200)
    assert np.allclose(G, la.inv(w), rtol=1e-12, atol=1e-12)


def test_scalar_case_matches_scaled_scalar_cauchy():
    # n=1: E[(w - beta X)^(-1)] = (1/beta) * G_sc(w/beta)
    omega = _omega()
    beta = 2.0  # real nonzero
    w = np.array([[omega]], dtype=np.complex128)
    b = np.array([[beta]], dtype=np.complex128)

    G = matrix_cauchy_semicircle_reference(w, b, n_quad=600)[0, 0]
    expected = (1.0 / beta) * semicircle_cauchy_scalar(omega / beta)

    assert abs(G - expected) < 5e-8


def test_b_identity_reduces_to_scalar_times_identity():
    n = 4
    omega = _omega()
    w = omega * np.eye(n, dtype=np.complex128)
    b = np.eye(n, dtype=np.complex128)

    G = matrix_cauchy_semicircle_reference(w, b, n_quad=600)
    expected = semicircle_cauchy_scalar(omega) * np.eye(n, dtype=np.complex128)

    assert np.allclose(G, expected, rtol=1e-10, atol=1e-10)


def test_diagonal_b_matches_entrywise_formula():
    omega = _omega()
    lam = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    n = lam.size

    w = omega * np.eye(n, dtype=np.complex128)
    b = np.diag(lam).astype(np.complex128)

    G = matrix_cauchy_semicircle_reference(w, b, n_quad=800)

    diag_expected = np.array(
        [(1.0 / l) * semicircle_cauchy_scalar(omega / l) for l in lam],
        dtype=np.complex128,
    )
    expected = np.diag(diag_expected)

    assert np.allclose(G, expected, rtol=1e-9, atol=1e-9)


def test_conjugation_symmetry_for_real_b_when_w_scalar_times_I():
    # Для вещественной плотности: G(\bar{omega}) = G(omega)^*
    omega = _omega()
    n = 3
    w = omega * np.eye(n, dtype=np.complex128)

    # Возьмём вещественный симметричный b
    b = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 2.0],
                  [0.0, 2.0, -1.0]], dtype=np.complex128)

    G = matrix_cauchy_semicircle_reference(w, b, n_quad=600)
    w_conj = np.conjugate(omega) * np.eye(n, dtype=np.complex128)
    G_conj = matrix_cauchy_semicircle_reference(w_conj, b, n_quad=600)

    assert np.allclose(G_conj, G.conj().T, rtol=1e-9, atol=1e-9)


def test_convergence_in_n_quad_sanity():
    # Не “теорема”, а sanity: при увеличении n_quad ответ стабилизируется
    omega = _omega()
    n = 3
    w = omega * np.eye(n, dtype=np.complex128)
    b = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 2.0]], dtype=np.complex128)

    G1 = matrix_cauchy_semicircle_reference(w, b, n_quad=200)
    G2 = matrix_cauchy_semicircle_reference(w, b, n_quad=800)

    # Относительная разница должна быть небольшой
    denom = max(1e-14, la.norm(G2, ord="fro"))
    rel = la.norm(G2 - G1, ord="fro") / denom
    assert rel < 5e-6
