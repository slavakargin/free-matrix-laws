# How to: anticommutator densities via subordination

This page shows how to compute the eigenvalue density of the anticommutator
$p(X,Y) = XY + YX$ of two free random variables using the **subordination**
method of Belinschi–Mai–Speicher, as implemented by
`subordination_kronecker`. Unlike the linearization solver
`polynomial_density`, the subordination route works for **any** pair of scalar
laws $X,Y$ (semicircle, free Poisson, custom), not only semicircular ones.

## Background

Self-adjoint polynomials in two free variables can be linearized as

$$
L_p \;=\; a_0 + a_1 \otimes X + a_2 \otimes Y,
$$

with fixed $n\times n$ matrices $a_0, a_1, a_2$. For the anticommutator
$p(X,Y) = XY + YX$ the standard $3\times 3$ linearization is

$$
a_0 =
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{bmatrix},
\quad
a_1 =
\begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix},
\quad
a_2 =
\begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{bmatrix}.
$$

The matrix variables $a_1 \otimes X$ and $a_2 \otimes Y$ are freely independent,
so $L_p - a_0$ is a free additive convolution. Each summand has a matrix Cauchy
transform $G_{a_i \otimes X}(w) = \mathbb{E}\big[(w - a_i\otimes X)^{-1}\big]$ and an
associated **subordination $h$-function**

$$
h_X(w) \;=\; G_{a_1 \otimes X}(w)^{-1} - w,
\qquad
h_Y(w) \;=\; G_{a_2 \otimes Y}(w)^{-1} - w .
$$

The subordination function $\omega_1(b)$ is the fixed point of the map

$$
w \;\longmapsto\; h_Y\!\big(h_X(w) + b\big) + b,
$$

evaluated at the regularized driving matrix

$$
b \;=\; \Lambda_\varepsilon(z) - a_0,
\qquad
\Lambda_\varepsilon(z) =
\begin{bmatrix} z & 0 \\ 0 & i\varepsilon\, I_{n-1} \end{bmatrix}.
$$

Once $\omega_1(b)$ is found, the Cauchy transform of the sum is recovered from the
first summand,

$$
G_{X+Y}(b) \;=\; G_{a_1 \otimes X}\!\big(\omega_1(b)\big),
$$

and the scalar density at a real point $x$ follows by Stieltjes inversion of the
distinguished corner, with $z = x + i\varepsilon$:

$$
f(x) \;\approx\; -\frac{1}{\pi}\,\Im\, \big[G_{X+Y}(b)\big]_{11}.
$$

## What you need

1. The linearization matrices $a_0, a_1, a_2$ for your polynomial.
2. A scalar Cauchy transform for each variable, passed as a callable
   `cauchy_scalar(z) -> complex`. The package ships
   `semicircle_cauchy_scalar` and `free_poisson_cauchy_scalar`; any function with
   the right signature works (use `functools.partial` to fix parameters such as
   the free Poisson rate $\lambda$).

## API

### `subordination_kronecker(b, a1, a2, cauchy_scalar_x=..., cauchy_scalar_y=..., eps=1e-4, tol=1e-8)`

Iterates the map $w \mapsto h_Y(h_X(w) + b) + b$ to its fixed point $\omega_1(b)$.
The defaults use the semicircle law for both variables. Because linearization
matrices $a_1, a_2$ are typically rank-deficient, the internal $h$-functions
regularize via $a \mapsto a + i\varepsilon I$; the default `eps=1e-4` is larger
than the `cauchy_kronecker` default for this reason, and the achievable accuracy
is $O(\varepsilon)$.

### `cauchy_kronecker(w, a, cauchy_scalar=..., eps=1e-8)`

Computes $G_{a\otimes X}(w) = \mathbb{E}\big[(w - a\otimes X)^{-1}\big]$ for any scalar
law, used both inside the $h$-functions and to recover $G_{X+Y}$ from
$\omega_1(b)$.

### `lambda_eps(z, n, eps=1e-6, block_size=1)`

Builds the regularized spectral parameter $\Lambda_\varepsilon(z)$.

## Example: anticommutator of two free semicircles

```python
import numpy as np
from free_matrix_laws import (
    subordination_kronecker, cauchy_kronecker, lambda_eps,
)

A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

def density_anticommutator(x, eps=0.01):
    n = 3
    z = x + eps * 1j
    b = lambda_eps(z, n) - A0
    omega = subordination_kronecker(b, A1, A2)          # semicircle by default
    G = cauchy_kronecker(omega, A1)
    return -G[0, 0].imag / np.pi

print(density_anticommutator(0.5))
```

This matches `polynomial_density(x, A0, np.array([A1, A2]))` for the same problem.

## Example: anticommutator of two free Poisson variables

Pass the free Poisson scalar transform for both variables, fixing the rate
$\lambda$ with `functools.partial`:

```python
from functools import partial
from free_matrix_laws import free_poisson_cauchy_scalar

def density_anticommutator_fpoisson(x, lam, eps=0.01):
    n = 3
    z = x + eps * 1j
    b = lambda_eps(z, n) - A0
    Gp = partial(free_poisson_cauchy_scalar, lam=lam)
    omega = subordination_kronecker(b, A1, A2, cauchy_scalar_x=Gp, cauchy_scalar_y=Gp)
    G = cauchy_kronecker(omega, A1, cauchy_scalar=Gp)
    return -G[0, 0].imag / np.pi
```

The resulting density integrates to $1$ with mean $2\lambda^2$
(since $\mathbb{E}[XY+YX] = 2\,\mathbb{E}[X]\,\mathbb{E}[Y] = 2\lambda^2$).

## Example: mixed semicircle and free Poisson

Use a different scalar transform for each variable. The recovery step uses the
transform of the **first** variable (here the semicircle, attached to `A1`):

```python
from free_matrix_laws import semicircle_cauchy_scalar

def density_anticommutator_mixed(x, lam, eps=0.01):
    n = 3
    z = x + eps * 1j
    b = lambda_eps(z, n) - A0
    Gp = partial(free_poisson_cauchy_scalar, lam=lam)
    omega = subordination_kronecker(
        b, A1, A2,
        cauchy_scalar_x=semicircle_cauchy_scalar,   # X (a1) is semicircular
        cauchy_scalar_y=Gp,                         # Y (a2) is free Poisson
    )
    G = cauchy_kronecker(omega, A1, cauchy_scalar=semicircle_cauchy_scalar)
    return -G[0, 0].imag / np.pi
```

Because the semicircle is centered, this anticommutator density is symmetric
about $0$, with mean $2\,\mathbb{E}[X]\,\mathbb{E}[Y] = 0$.

## Tips

- Evaluate at $z = x + i\varepsilon$ with $\varepsilon \approx 10^{-2}$; the
  edge smoothing it induces can produce tiny negative density values near the
  tails, which is expected.
- Keep the solver's `eps` consistent between `subordination_kronecker` and the
  final `cauchy_kronecker` recovery; the default `1e-4` is a good starting point.
- The fixed-point iteration typically converges in a few tens of steps. If it
  stalls, raise `eps` slightly or increase `maxiter`.
- To verify a result, compare against a random-matrix simulation: build large
  random samples of $X$ and $Y$ and histogram the eigenvalues of $XY + YX$.