# Solving the operator-valued Cauchy transform $G(z)$

This shows how to compute the matrix Cauchy transform $G(z)$ for the **matrix semicircle** via the fixed-point solver
$$
z\,G \;=\; I \;+\; \eta(G)\,G,\qquad \Im z>0,\quad
\eta(B)=\sum_{i=1}^s A_i\,B\,A_i^\ast.
$$

## Minimal example

```python
import numpy as np
import free_matrix_laws as fml

n, s = 5, 3
rng = np.random.default_rng(0)

# Kraus operators A_i (general complex case)
A = [rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)) for _ in range(s)]

z = 0.3 + 1j*0.02  # Im z > 0
G = fml.solve_cauchy_semicircle(z, A, tol=1e-10, maxiter=2000)

# Residual check: || zG - I - eta(G)G || should be small
etaG = fml.covariance_map(G, A)
res = np.linalg.norm(z*G - np.eye(n) - etaG @ G)
print("residual:", res)
```

### Scalar reduction sanity check

When all $A_i=\sigma I$, the solution is $G(z)=g(z),I$ with the scalar semicircle Stieltjes transform
$$
g(z)=\frac{z-\sqrt{z^2-4c}}{2c},\qquad c=\sigma^2.
$$

```python
import numpy as np
import free_matrix_laws as fml

n, sigma = 6, 1.3
A = [sigma*np.eye(n)]
c = sigma**2

z = -0.5 + 1j*0.05
G = fml.solve_cauchy_semicircle(z, A, tol=1e-12)

g = (z - np.sqrt(z*z - 4*c)) / (2*c)
err = np.linalg.norm(G - g*np.eye(n))
print("||G - gI||:", err)
```

### Tips

* **Imaginary part of $z$:** For densities at real $x$, use $z = x + i,\varepsilon$ with $\varepsilon \approx 10^{-2}\text{–}10^{-3}$.
* **Warm starts:** Pass `G0` (e.g., the solution at a nearby $z$) to accelerate convergence.
* **Input shapes:** $A$ can be a list/tuple of $(n,n)$ arrays or a stacked array of shape $(s,n,n)$.
* **Numerical stability:** If the iteration stalls, increase $\varepsilon$, relax `tol`, or cap `maxiter`.

### From $G(z)$ to a scalar Stieltjes transform and density

The scalar transform is
$$
m(z) = \frac{1}{n} \mathrm{tr} \, G(z).
$$
For $x\in\mathbb{R}$ with $z=x+i\varepsilon$,
$$
f(x) = -\frac{1}{\pi}\Im m(z).
$$

```python
import numpy as np
import free_matrix_laws as fml

n = 5
A = [np.eye(n)]           # simple case
x, eps = 0.0, 1e-2
z = x + 1j*eps

G = fml.solve_cauchy_semicircle(z, A, tol=1e-12)
m = np.trace(G)/n
f = (-1/np.pi) * np.imag(m)
print("density ~", f)
```

# Or use the convenience helper:

```python
f2 = fml.semicircle_density(x, A, eps=eps, tol=1e-12)
print("density (helper) ~", f2)
```
