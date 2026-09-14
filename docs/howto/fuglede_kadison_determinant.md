# How to: Fuglede–Kadison determinant of the shifted matrix semicircle

This page shows how to compute the (log) Fuglede–Kadison determinant of the
shifted matrix semicircle

$$
S \;=\; a_0 + \sum_{i=1}^r A_i \otimes X_i,
$$

where the $X_i$ are free standard semicircular elements and the coefficients
$A_i = A_i^\ast$ and mean $a_0 = a_0^\ast$ are fixed Hermitian $m\times m$
matrices. The function is `biased_matrix_semicircle_logdet` (short alias
`bms_logdet`). It reuses the Dyson solver `cauchy_biased_matrix_semicircle`, so
one matrix solve plus a little algebra gives the answer.

## Background

The Fuglede–Kadison determinant, taken with respect to
$\operatorname{tr}_m \otimes \tau$ (with $\operatorname{tr}_m = \tfrac1m\operatorname{Tr}$),
is the logarithmic potential of the spectral measure $\mu_S$:

$$
\log\Delta(S - x) \;=\; \int_{\mathbb R} \log|t - x|\, d\mu_S(t),
\qquad x\in\mathbb R.
$$

A global spectral quantity like this could in principle be obtained by first
computing the density $\mu_S$ and then integrating $\log|t-x|$ against it. That
route is expensive and, for $x$ inside the support, the integrand is singular.
Instead we use a **closed formula** that reads the determinant directly off the
single matrix $G$ solving the Dyson equation.

Let $G = G(z)$ solve

$$
G^{-1} + \eta(G) = z I - a_0,
\qquad
\eta(B) = \sum_{i=1}^r A_i B A_i^\ast,
\qquad \Im z > 0,
$$

which is exactly what `cauchy_biased_matrix_semicircle(z, a0, A)` returns. Then

$$
\boxed{\;
\log\Delta(S - x)
\;=\;
\operatorname{Re}\!\left[
  -\frac1m \log\det G
  + \frac12\, \operatorname{tr}_m\!\big(G\,\eta(G)\big)
\right]
\;}
$$

evaluated at $z = x + i\varepsilon$. Three features make this convenient:

- **Only the real part enters.** Two holomorphic branches of $\log\det G$ differ
  by $2\pi i k / m$, which is purely imaginary, so the branch is irrelevant and
  the modulus form $-\tfrac1m\log|\det G|$ is used directly (via `slogdet`).
- **No optimization or constraints.** This is a proven identity, not the
  variational characterization; it needs one Dyson solve, not a minimization
  over a feasible set of positive matrices.
- **Natural regularization.** For $x$ in an unbounded component of the resolvent
  set, $G$ is self-adjoint and $\varepsilon$ barely matters. For $x$ inside the
  support, the offset makes the boundary value $G(x+i0)$ well defined, and the
  result approaches $\log\Delta(S-x)$ as $\varepsilon\downarrow0$ (bulk error
  $O(\varepsilon^2)$, edge error $O(\varepsilon)$).

## What you need

1. The mean matrix $a_0 = a_0^\ast$ (shape `(m, m)`).
2. The Hermitian coefficients `A = (A_1, ..., A_r)` (a list of `(m, m)` arrays or
   a stacked `(r, m, m)` array) defining $\eta$.
3. A real evaluation point $x$.

## API

### `biased_matrix_semicircle_logdet(x, a0, A, eps=1e-4, G0=None, tol=1e-12, maxiter=5000)`

Returns $\log\Delta(S - x)$ as a Python `float`. Exponentiate to get
$\Delta(S - x)$ itself. `bms_logdet` is a short alias for the same function.
Internally it calls `cauchy_biased_matrix_semicircle` at $z = x + i\varepsilon$
and applies the boxed formula; `eta` supplies $\eta(G)$ for the trace term.

## Example: scalar semicircle (a sanity check)

For $m = 1$ the object is $S = a + \sigma s$ with $s$ standard semicircular,
supported on $[a - 2\sigma,\, a + 2\sigma]$. The formula reproduces the classical
logarithmic potential of the semicircle. Inside the support,

$$
\log\Delta(S - x) \;=\; \log\sigma - \tfrac12 + \frac{(a - x)^2}{4\sigma^2},
\qquad |a - x| \le 2\sigma .
$$

```python
import numpy as np
from free_matrix_laws import biased_matrix_semicircle_logdet as bms_logdet

sigma, a = 1.0, 0.0
A = np.array([[[sigma]]])     # single 1x1 coefficient
a0 = np.array([[a]])

# outside the support (|a - x| > 2 sigma):
print(bms_logdet(3.0, a0, A))     # -> 1.0354...

# inside the support, compare with the exact potential:
x = 1.0
exact = np.log(sigma) - 0.5 + (a - x)**2 / (4 * sigma**2)
print(bms_logdet(x, a0, A), exact)   # -> -0.2499...  -0.25
```

## Example: matrix semicircle

```python
import numpy as np
from free_matrix_laws import biased_matrix_semicircle_logdet as bms_logdet

A = np.array([[[1.0, 0.2], [0.2, 0.5]],
              [[0.0, 0.4], [0.4, -0.3]]])   # two 2x2 Hermitian coefficients
a0 = np.array([[0.3, 0.1], [0.1, -0.2]])

xs = np.linspace(-6, 6, 25)
logdet = np.array([bms_logdet(x, a0, A) for x in xs])
```

Each value equals $\int \log|t - x|\, d\mu_S(t)$ for the spectral law of
$S = a_0 + A_1\otimes X_1 + A_2\otimes X_2$.

## Tips

- **Choosing $\varepsilon$.** The default `eps=1e-4` is a good balance. Outside
  the support you can push it much smaller with no cost to convergence; inside
  the support a smaller $\varepsilon$ sharpens the result but slows the Dyson
  iteration. Increase `maxiter` if the solver does not reach `tol`.
- **Getting $\Delta$ itself.** The function returns the logarithm; use
  `np.exp(bms_logdet(...))` for the determinant.
- **Cross-checks.** Outside the support the integrand $\log|t-x|$ is smooth, so
  $\int \log|t-x|\,\rho(t)\,dt$ with $\rho$ from
  `biased_matrix_semicircle_density` is a clean, independent check. For a fully
  independent verification at any $x$, build large random samples of $S$ and
  average $\tfrac{1}{mN}\sum_k \log|\lambda_k - x|$ over the eigenvalues.
