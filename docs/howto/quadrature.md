# How-to: Quadrature-based matrix Cauchy transform

This guide shows how to compute the **matrix-valued Cauchy (Stieltjes)
transform** of $b \otimes X$ using brute-force numerical integration,
and how to cross-validate it against the fixed-point solver.

## Background

Given a deterministic $n \times n$ matrix $b$ and a standard semicircular
random variable $X$ (i.e. with Wigner semicircle distribution
$\mathrm{d}\mu_{\mathrm{sc}}(x) = \frac{1}{2\pi}\sqrt{4-x^2}\,\mathrm{d}x$
on $[-2,2]$), the **matrix-valued Cauchy transform** is

$$
    G_b(w) = \int_{-2}^{2} (w - x\,b)^{-1}\,\mathrm{d}\mu_{\mathrm{sc}}(x),
$$

where $w$ is an $n\times n$ matrix spectral parameter (typically
$w = (x_0 + i\varepsilon)I_n$ for Stieltjes inversion).

The **quadrature module** (`free_matrix_laws.quadrature`) evaluates this
integral entry-by-entry via `scipy.integrate.quad`, using the identity

$$
    G_b(w) = -\frac{1}{\pi}\int_{-\infty}^{\infty}
        (w - z\,b)^{-1}\,\operatorname{Im}\!\big[G_{\mathrm{sc}}(z)\big]\,\mathrm{d}x,
    \qquad z = x + i\varepsilon,
$$

where $G_{\mathrm{sc}}(z) = \frac{z - \sqrt{z^2 - 4}}{2}$ is the scalar
semicircle Cauchy transform.

## Quick example

```python
import numpy as np
from free_matrix_laws import cauchy_matrix_semicircle, h_matrix_semicircle

# Coefficient matrix
b = np.array([[1.0, 0.2],
              [0.2, 0.8]])

# Spectral parameter w = z * I  with z = 0.5 + 0.1i
w = (0.5 + 0.1j) * np.eye(2)

# Matrix-valued Cauchy transform (brute-force)
G = cauchy_matrix_semicircle(w, b, eps=1e-3)
print("G_b(w) =")
print(G)

# h-function:  h(w) = G(w)^{-1} - w
h = h_matrix_semicircle(w, b, eps=1e-3)
print("\nh_b(w) =")
print(h)
```

## Recovering the scalar density

To get the scalar spectral density of $b \otimes X$ at a real point $x$,
use `density_scalar_quadrature`:

```python
import numpy as np
from free_matrix_laws import density_scalar_quadrature

b = np.eye(3)

# Evaluate at several points
xs = np.linspace(-3, 3, 61)
ds = [density_scalar_quadrature(x, b, eps=1e-2) for x in xs]
```

For $b = I_n$ this reproduces the standard Wigner semicircle law (up to
the regularization parameter `eps`).

## Cross-validating with the fixed-point solver

The quadrature approach is independent of the fixed-point iteration in
`solve_cauchy_semicircle`.  This makes it a valuable cross-check:

```python
import numpy as np
from free_matrix_laws import semicircle_density, density_scalar_quadrature

b = np.array([[1.0, 0.2],
              [0.2, 0.8]])
A = [b]  # single Kraus operator

x = 0.5
eps = 1e-2

f_fp   = semicircle_density(x, A, eps=eps, tol=1e-11, maxiter=5000)
f_quad = density_scalar_quadrature(x, b, eps=eps)

print(f"Fixed-point solver:  {f_fp:.8f}")
print(f"Quadrature:          {f_quad:.8f}")
print(f"Difference:          {abs(f_fp - f_quad):.2e}")
```

## When to use this vs. the fixed-point solver

| | Quadrature (`cauchy_matrix_semicircle`) | Fixed-point (`solve_cauchy_semicircle`) |
|---|---|---|
| **Speed** | $O(n^5 Q)$ — slow for large $n$ | $O(n^3 K)$ — much faster |
| **Generality** | Single matrix $b$ | Arbitrary Kraus maps $\eta$ |
| **Use case** | Testing / validation | Production |
| **Dependencies** | Requires `scipy` | Pure `numpy` |

## Parameters that matter

- **`eps`** (imaginary shift): Controls regularization.  Smaller values give
  sharper densities but may cause near-singular resolvents.  Typical range:
  $10^{-3}$ to $10^{-2}$.
- **`x_min`, `x_max`** (integration limits): Default $[-4, 4]$ is generous.
  The semicircle is supported on $[-2, 2]$, so anything beyond that suffices.
- **`quad_opts`**: Forwarded to `scipy.integrate.quad` for fine-tuning
  (e.g. `{'limit': 200, 'epsabs': 1e-12}`).
