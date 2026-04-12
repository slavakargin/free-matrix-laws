# How to detect and classify singularities at zero

Given a matrix semicircular variable $S = \sum_i A_i \otimes s_i$, the scalar
spectral density $f(x)$ may diverge at $x = 0$.  Near such a singularity the
density behaves as
$$
f(x) \;\sim\; C\,|x|^{\alpha}, \qquad x \to 0,
$$
where $\alpha < 0$ (integrable cusp) or $\alpha = 0$ (logarithmic or
non-singular).  The exponent $\alpha$ is a **Puiseux exponent** of the HMS
subordination function $W(u)$ at $u = 0$: since
$f(0) = \frac{1}{\pi}\operatorname{tr}(\Re\,W(0^+))$,
a divergence in $\operatorname{tr} W(u) \sim C\,u^{\alpha}$ translates directly
to the density.

`singularity_report` automates the detection: it solves the HMS equation
$$
\eta(W)\,W + u\,W = I
$$
on a fine geometric grid of $u$ values approaching zero, then estimates
$\alpha$ by log–log regression.

## Prerequisites

```python
import numpy as np
from free_matrix_laws import singularity_report
```

## Quick start: one-call analysis

```python
# Pencil with a cusp singularity (known exponent -1/3)
A1 = np.array([[0, 1], [1, 0]])
A2 = np.array([[0, 0], [0, 1]])

result = singularity_report([A1, A2])
```

This prints a summary like:

```
============================================================
  Singularity report for HMS equation at u = 0
============================================================
  Matrix size:          2 x 2
  Grid:                 300 points, u in [4.40e-08, 5.00e-01]
  Converged:            275 / 300

  SINGULAR at u = 0
    tr W(u) ~ 0.5012 * u^(-0.3332)   [R² = 1.000000]
    Nearest simple fraction: -1/3  (= -0.333333)
    => density f(x) ~ |x|^(-0.3333) near x = 0

  Eigenvalue exponents of Re W(u):
    eig_1:  alpha = -0.3334
    eig_2:  alpha = 0.3332
  det(Re W) exponent:  alpha = -0.0002   [R² = 0.708683]
============================================================
```

## Non-singular case

```python
# Standard semicircle: A = I (doubly stochastic)
result = singularity_report([np.eye(2)])
```

Output:

```
  NON-SINGULAR at u = 0
    tr W(0+) ≈ 0.999967
    => f(0) = 0.318299
```

The value $f(0) = 1/\pi \approx 0.3183$ matches the Wigner semicircle.

## Programmatic access

The returned dictionary contains all computed data:

```python
result = singularity_report([A1, A2], verbose=False)

result["singular"]     # True/False
result["alpha_tr"]     # Puiseux exponent of tr W(u)
result["C_tr"]         # prefactor: tr W(u) ~ C * u^alpha
result["R2_tr"]        # R² of the log-log fit
result["alpha_eigs"]   # array of exponents per eigenvalue
result["alpha_det"]    # exponent of det(Re W)
result["n_converged"]  # how many grid points converged

# Raw sweep data
sweep = result["sweep"]
sweep["u"]             # (m,) array of u values
sweep["tr_W"]          # (m,) normalized trace
sweep["eigs"]          # (m, n) eigenvalues of Re W
sweep["det_W"]         # (m,) determinant of Re W
sweep["converged"]     # (m,) convergence flags
```

## Using the Newton solver directly

The module also exposes `solve_hms`, a Newton solver for the HMS equation
that converges much faster than the standard half-averaged fixed-point
iteration, especially near singularities:

```python
from free_matrix_laws import solve_hms

W, iters, residual = solve_hms(u=0.01, A=[A1, A2])
# W satisfies eta(W) W + 0.01 W = I  up to residual ~ 1e-14
```

Warm-starting from a nearby solution is recommended when sweeping:

```python
W_prev = None
for u in [0.1, 0.05, 0.01, 0.005, 0.001]:
    W, iters, res = solve_hms(u, [A1, A2], W0=W_prev)
    W_prev = W.copy()
    print(f"u={u:.3f}  iters={iters}  res={res:.2e}")
```

## Congruence invariance

The trace exponent $\alpha$ is invariant under congruence $S' = cSc^*$
(equivalently, $A_i' = cA_ic^*$).  Individual eigenvalue exponents of $W$ are
**not** invariant — a unitary congruence can swap them:

```python
# Original
r1 = singularity_report([A1, A2], verbose=False)

# Congruent via the swap matrix c = A1
A2_swap = np.array([[1, 0], [0, 0]])
r2 = singularity_report([A1, A2_swap], verbose=False)

print(f"alpha_tr:   {r1['alpha_tr']:.4f} vs {r2['alpha_tr']:.4f}")  # same
print(f"alpha_eigs: {r1['alpha_eigs']} vs {r2['alpha_eigs']}")       # swapped
```

## Tips

* **Grid density matters.** The default 300 points is usually fine.  For
  difficult cases (slow convergence near $u = 0$), increase `n_points` or
  raise `u_min`.
* **Convergence ceiling.**  The Newton solver may stall for very small $u$
  near a singularity (the Jacobian becomes ill-conditioned as $W$ degenerates).
  The report uses only well-converged points for the fit, so this is handled
  gracefully.
* **Rational approximation.**  The report uses `Fraction.limit_denominator(12)`
  to identify the nearest simple fraction — useful for recognizing exponents
  like $-1/3$, $-1/2$, $-1/4$ that arise from low-degree algebraic equations.

## See also

* API: [singularity (full signatures)](../api/singularity.md)
* How-to: [Semicircle density](semicircle_density.md) — computing the density itself
* How-to: [CP Sinkhorn](cp_sinkhorn.md) — DS-scalability (non-singularity criterion)