# How to: density for a polynomial in semicircular variables (via linearization)

This page documents the helper iteration step `hfsc_map` and the convenience wrapper
`polynomial_semicircle_density` (short aliases are `polynomial_density` and `get_density_C`), used to approximate the
density of a **self-adjoint polynomial** in free semicircular variables after you have
built a **self-adjoint linearization**.


## Background (what these functions compute)

Let $p$ be a self-adjoint polynomial in semicircular variables $X_1,\dots,X_s$.
Assume you already have a self-adjoint linearization $L_p$ of $p$ in the form

$$
L_p \;=\; a_0 + \sum_{j=1}^s a_j X_j,
$$

where $a_0,a_1,\dots,a_s$ are fixed complex matrices (of the same size).

To recover the scalar Cauchy transform of $p$ from the linearization, one considers the
regularized block-diagonal matrix

$$
\Lambda_\varepsilon(z)
=
\begin{bmatrix}
z & 0 & \cdots & 0 \\
0 & i\varepsilon & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & i\varepsilon
\end{bmatrix},
\qquad \varepsilon>0,
$$

and the *quasi-resolvent*

$$
F_\varepsilon(z) \;=\; \big(\Lambda_\varepsilon(z) - L_p\big)^{-1}.
$$

The scalar Cauchy transform of $p$ is then obtained from the $(1,1)$ entry/block via
the limit $\varepsilon\downarrow 0$; numerically, we use a small $\varepsilon$ and
approximate

$$
G_p(z) \;\approx\; \big[F_\varepsilon(z)\big]_{1,1}.
$$

Finally, the density at a real point $x$ is obtained by Stieltjes inversion:
for $z=x+i\varepsilon$,

$$
f(x) \;\approx\; -\frac{1}{\pi}\,\Im\,G_p(x+i\varepsilon).
$$

## What you need to provide

These functions assume you already know:

1. The **linearization bias** matrix $a_0$ (called `a` below).
2. The list/stack of **Kraus matrices** `AA = (A_1,\dots,A_s)` that define the covariance map
   $$
   \eta(B) = \sum_{i=1}^s A_i\,B\,A_i^\ast.
   $$

> In typical linearizations, the $A_i$ are built from the coefficient matrices $a_j$
> of the linearization, but this construction is outside the scope of this how-to.

## API

### `_hfsc_map(G, z, a0, A)`

This is a single fixed-point step used inside an iteration scheme.

Conceptually, it implements a half-averaged update

$$
G \;\mapsto\; \frac12\Big[G + W(G)\Big],
$$

where $W(G)$ is the “raw” update derived from the linearized equation and the map $\eta$.

### `polynomial_semicircle_density(x, a0, A, eps=1e-2, ...)`

This computes the density at a real point $x$ by:
1) solving for $G(z)$ at $z=x+i\,\varepsilon$ by iteration, and
2) returning $-(1/\pi)\Im(G(z)_{11})$.

## Example: anticommutator $X_1X_2 + X_2X_1$ via a $3\times 3$ linearization

A standard self-adjoint linearization for the anticommutator is the matrix

$$
S
=
\begin{bmatrix}
0 & X_1 & X_2 \\
X_1 & 0 & -1 \\
X_2 & -1 & 0
\end{bmatrix}
=
a_0 + a_1 X_1 + a_2 X_2,
$$

with

$$
a_0 =
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0
\end{bmatrix},
\quad
a_1 =
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix},
\quad
a_2 =
\begin{bmatrix}
0 & 0 & 1 \\
0 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}.
$$

Once you have built the corresponding Kraus family `A` for $\eta$, you can evaluate the
density numerically:

```python
import numpy as np
from free_matrix_laws.transforms import polynomial_semicircle_density

# linearization bias a0:
a0 = np.array([
    [0, 0, 0],
    [0, 0,-1],
    [0,-1, 0],
], dtype=complex)

# A = (A1,...,As) should be prepared for your problem.
# Here we assume you already have it:
A = ...  # list of (n,n) arrays or stacked array (s,n,n)

x = 0.0
fx = polynomial_semicircle_density(x, a0, A, eps=1e-2, maxiter=10_000)
print("f(x) ≈", fx)
```

## Tips

- Use $z=x+i\varepsilon$ with $\varepsilon\approx 10^{-2}$–$10^{-3}$.
- If the iteration stalls, increase $\varepsilon$, relax `tol`, or reduce `maxiter`.
- Warm-starting (re-using the solution for a nearby $z$) can speed up convergence a lot.
- The returned density is a numerical approximation; decreasing $\varepsilon$ sharpens it but may require tighter tolerances.

## What is actually returned

`polynomial_semicircle_density(x, ...)` returns
$$
f(x)\;\approx\;-\frac{1}{\pi}\,\operatorname{Im}\,G(x+i\varepsilon)_{11}.
$$

In many linearization setups, the $(1,1)$ block corresponds exactly to the scalar resolvent
of the polynomial. If your linearization uses a larger $(1,1)$ block, replace $G_{11}$
by the normalized trace of that block.
