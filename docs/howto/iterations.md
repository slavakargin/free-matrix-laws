# Fixed-point iterations: how the solvers work

This page explains the iterative schemes used internally by the Cauchy-transform
solvers. Understanding the single-step maps is useful for debugging convergence,
tuning relaxation, or extending the package to new settings.

---

## Unbiased case: `_hfs_map`

For the matrix semicircle $S = \sum_i A_i \otimes X_i$, the Cauchy transform
$G(z)$ satisfies the **Speicher equation**

$$
z\,G \;=\; I \;+\; \eta(G)\,G,
\qquad \eta(B) = \sum_{i=1}^s A_i\,B\,A_i^*.
$$

The raw Picard step would be $G \mapsto (zI - \eta(G))^{-1}$, but it can
oscillate. The **half-averaged** (Helton–Rashidi Far–Speicher) step damps
the update:

$$
T(G) \;=\; \tfrac{1}{2}\Big[\,G \;+\; (zI - \eta(G))^{-1}\,\Big].
$$

This is implemented as `_hfs_map(G, z, A)` in `transforms.py` and is the
building block of [`cauchy_matrix_semicircle`][free_matrix_laws.cauchy_matrix_semicircle].

### Example: watching convergence

```python
import numpy as np, numpy.linalg as la
from free_matrix_laws.transforms import _hfs_map
from free_matrix_laws import covariance_map as eta

rng = np.random.default_rng(0)
n, s = 3, 2
A = [rng.standard_normal((n, n)) for _ in range(s)]
z = 0.5 + 1.0j

G = -1j * np.eye(n)
for k in range(200):
    G_new = _hfs_map(G, z, A)
    diff = la.norm(G_new - G)
    if diff < 1e-12:
        print(f"Converged at step {k}")
        break
    G = G_new

# Verify Speicher equation
R = z * G - np.eye(n) - eta(G, A) @ G
print(f"Residual: {la.norm(R):.2e}")
```

---

## Biased case: `_hfsb_map`

For the biased matrix semicircle $S = a_0 + \sum_i A_i \otimes X_i$,
the Cauchy transform satisfies

$$
G \;=\; (zI - a_0 - \eta(G))^{-1}
\quad\Longleftrightarrow\quad
(zI - a_0)\,G \;=\; I + \eta(G)\,G.
$$

Setting $b = z\,(zI - a_0)^{-1}$, the fixed-point map becomes

$$
\Phi(G) \;=\; \big[\,zI - b\,\eta(G)\,\big]^{-1}\,b,
$$

and the **relaxed** (half-averaged) step is

$$
G_{\text{new}} \;=\; (1 - \alpha)\,G \;+\; \alpha\,\Phi(G),
$$

where $\alpha \in (0,1]$ is the relaxation parameter (default $\alpha = 0.5$).

This is implemented as `_hfsb_map(G, z, a0, A, relax=0.5)` and is the
core step of [`cauchy_biased_matrix_semicircle`][free_matrix_laws.cauchy_biased_matrix_semicircle].

### Why the $b$-substitution?

The substitution $b = z(zI - a_0)^{-1}$ transforms the biased equation into
the same algebraic form as the unbiased one (with $b$ replacing $I$), so the
same half-averaging strategy applies. When $a_0 = 0$ we get $b = I$ and
the biased solver reduces exactly to the unbiased one.

### Convergence is monitored by the residual

$$
R(G) \;=\; (zI - a_0)\,G \;-\; I \;-\; \eta(G)\,G,
$$

and the solver stops when $\|R\|_F \le \text{tol}$.

### Example: manual iteration with `_hfsb_map`

```python
import numpy as np, numpy.linalg as la
from free_matrix_laws.transforms import _hfsb_map
from free_matrix_laws import covariance_map as eta

rng = np.random.default_rng(1)
n, s = 4, 2
a0 = rng.standard_normal((n, n))
a0 = (a0 + a0.T) / 2          # make Hermitian
A = [rng.standard_normal((n, n)) for _ in range(s)]
z = 0.3 + 0.05j

I = np.eye(n, dtype=complex)
G = -1j * I / np.imag(z)

for k in range(2000):
    G = _hfsb_map(G, z, a0, A, relax=0.5)
    R = (z * I - a0) @ G - I - eta(G, A) @ G
    if la.norm(R, 'fro') < 1e-12:
        print(f"Converged at step {k}")
        break

print(f"Residual: {la.norm(R, 'fro'):.2e}")
```

### Tuning the relaxation parameter

| `relax` | Behaviour |
|---------|-----------|
| 0.5     | Default; robust damping for most problems |
| 1.0     | No averaging (pure Picard); faster when it works, may diverge |
| < 0.5   | Stronger damping; use if 0.5 diverges (rare) |

---

## Linearized polynomial case: `_hfsc_map`

For a self-adjoint polynomial $p(X_1, \dots, X_s)$ with linearization
$L_p = a_0 + \sum_i A_i \otimes X_i$, the iteration uses the regularized
parameter $\Lambda_\varepsilon(z)$ (see [`lambda_eps`][free_matrix_laws.lambda_eps])
and follows the same half-averaged pattern with
$b_\varepsilon(z) = z(\Lambda_\varepsilon(z) - a_0)^{-1}$.

This is the building block of [`cauchy_polynomial`][free_matrix_laws.cauchy_polynomial].