# How to: `semicircle_density`

Compute the scalar density of a **matrix semicircle** distribution at a real point $x$ via Stieltjes inversion.

```python
import numpy as np
from free_matrix_laws import semicircle_density
```

## Minimal example
```python
n, s = 3, 2
A1 = np.eye(n)
A2 = 2*np.eye(n)
A  = [A1, A2]

x  = 0.0
fx = semicircle_density(x, A, eps=1e-2)   # ≈ density at x
fx
```

## Vector of points
```python
xs  = np.linspace(-6, 6, 400)
fxs = np.array([semicircle_density(x, A, eps=5e-3) for x in xs])
```

## Signature
```python
semicircle_density(
    x: float,
    A,                 # list/tuple of (n,n) or stacked (s,n,n)
    eps: float = 1e-2,
    G0=None,           # optional warm start for G(z)
    tol: float = 1e-10,
    maxiter: int = 10_000,
) -> float
```

## What it does

For $z=x+i\varepsilon$ it solves
$$
z\,G \;=\; I \;+\; \eta(G)\,G, \qquad
\eta(B)=\sum_{i=1}^s A_i\,B\,A_i^\ast,\ \Im z>0,
$$
then returns
$$
f(x) \;=\; -\frac{1}{\pi}\,\Im\!\left(\frac{1}{n}\,\mathrm{tr}\,G(x+i\varepsilon)\right).
$$

## Tips
- **Imaginary part:** for real $x$, use $z=x+i\varepsilon$ with $\varepsilon\approx 10^{-2}$–$10^{-3}$.
- **Warm starts:** pass `G0` (e.g., the solution at a nearby $z$) to speed up convergence.
- **Input shapes:** `A` may be a list/tuple of $(n,n)$ arrays or a stacked array of shape $(s,n,n)$.
- **Numerical stability:** if iterations stall, increase $\varepsilon$, relax `tol`, or raise `maxiter`.

## Scalar-reduction sanity check

If all $A_i=\sigma I$, then $G(z)=g(z)I$ and the scalar semicircle Stieltjes transform is
$$
g(z)=\frac{z-\sqrt{z^{2}-4c}}{2c}, \qquad c=\sigma^{2}.
$$
The density matches the semicircle on $[-2\sqrt{c},\,2\sqrt{c}]$.

> See also: [`biased_semicircle_density`](biased_semicircle_density.md).
