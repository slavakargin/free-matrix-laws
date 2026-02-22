import numpy as np
import numpy.linalg as la
from free_matrix_laws.transforms import _hfs_map

def test_hfs_map_converges_and_satisfies_equation():
    rng = np.random.default_rng(0)
    n, s = 3, 2
    A = [rng.standard_normal((n,n)) for _ in range(s)]
    z = 0.5 + 1.0j
    G = -1j * np.eye(n)
    for _ in range(300):
        G_next = _hfs_map(G, z, A)
        if la.norm(G_next - G) < 1e-12:
            break
        G = G_next
    # Check Speicher equation: zG = I + eta(G) G
    from free_matrix_laws import covariance_map as eta
    R = z * G - np.eye(n) - eta(G, A) @ G
    assert la.norm(R) < 1e-8

def test_hfs_map_rejects_non_upper_half_plane_z():
    n = 2
    A = [np.eye(n)]
    G = -1j * np.eye(n)
    try:
        _hfs_map(G, 1.0 - 0.1j, A)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for Im(z) <= 0"