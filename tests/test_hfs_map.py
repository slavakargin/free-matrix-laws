import numpy as np
import numpy.linalg as la
from free_matrix_laws.transforms import _hfs_map
from free_matrix_laws import covariance_map as eta

def test_hfs_map_converges_and_satisfies_equation():
    rng = np.random.default_rng(0)
    n, s = 3, 3
    # Small-norm Kraus ops → well-conditioned resolvent
    A = [0.2*(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))) for _ in range(s)]
    z = 2.0 + 1.0j

    G = -1j * np.eye(n)  # reasonable starting guess in upper half-plane
    for _ in range(300):
        G_next = _hfs_map(G, z, A)
        if la.norm(G_next - G) <= 1e-10 * (1 + la.norm(G)):
            G = G_next
            break
        G = G_next

    R = z*G - np.eye(n) - eta(G, A) @ G
    assert la.norm(R) < 1e-7

def test_hfs_map_rejects_non_upper_half_plane_z():
    n = 2
    A = [np.eye(n)]
    G = -1j*np.eye(n)
    try:
        _hfs_map(G, 1.0 - 0.1j, A)
    except ValueError as e:
        assert "Im(z) > 0" in str(e)
    else:
        raise AssertionError("Expected ValueError for Im(z) <= 0")
