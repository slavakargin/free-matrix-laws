# How to: `biased_semicircle_density`

Density for the **biased matrix semicircle** $S=a_0+\sum_{i=1}^s A_i\otimes X_i$, where $X_i$ are free semicirculars.

```python
import numpy as np
from free_matrix_laws import biased_semicircle_density
```

## Minimal example
```python
n = 3
a0 = np.array([[0,0,0],[0,0,-1],[0,-1,0]], dtype=float)
A1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
A2 = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=float)
A  = [A1, A2]

x  = 0.0
fx = biased_semicircle_density(x, a0, A, eps=1e-2)
fx
```

## Vector of points
```python
xs  = np.linspace(-5, 5, 400)
fxs = np.array([biased_semicircle_density(x, a0, A, eps=5e-3) for x in xs])
```

## Signature
```python
biased_semicircle_density(
    x: float,
    a0,                # (n,n) bias matrix
    A,                 # list/tuple of (n,n) or stacked (s,n,n)
    eps: float = 1e-2,
    G0=None,           # optional warm start for G(z)
    tol: float = 1e-10,
    maxiter: int = 10_000,
) -> float
```

## What it does

For $z=x+i\varepsilon$ it computes the Cauchy transform $G_{a_0+X}(z)$ using the Helton–Rashidi Far–Speicher half-averaged step adapted to the bias:
$$
b \;=\; z\,(zI-a_0)^{-1},\qquad
G \leftarrow \tfrac{1}{2}\Big[\,G + (zI - b\,\eta(G))^{-1} b\,\Big],
$$
then returns
$$
f(x) \;=\; -\frac{1}{\pi}\,\Im\!\left(\frac{1}{n}\,\mathrm{tr}\,G(x+i\varepsilon)\right).
$$

## Tips
- **Reduction:** if $a_0=0$, this reduces numerically to `semicircle_density`.
- **Conditioning:** if $(zI-a_0)^{-1}$ is ill-conditioned, increase $\varepsilon$ slightly.
- Use the same advice as for $\varepsilon$, `G0`, `tol`, and `maxiter`.

## Sanity check (reduction)
```python
rng = np.random.default_rng(0)
n, s = 4, 3
A = [rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)) for _ in range(s)]
a0 = np.zeros((n,n), dtype=complex)
x  = 0.3

from free_matrix_laws import semicircle_density
f1 = biased_semicircle_density(x, a0, A, eps=1e-2)
f2 = semicircle_density(x, A, eps=1e-2)
assert abs(f1 - f2) < 1e-6
```

> See also: [`semicircle_density`](semicircle_density.md).
