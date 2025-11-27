# Solving $G(z)$ for a biased matrix semicircle

We consider $S = a_0 + \sum_{i=1}^s A_i \otimes X_i$ with semicircular $X_i$ and covariance
$\eta(B)=\sum_i A_i B A_i^\ast$. The Cauchy transform solves
$$
G(z) \;=\; (z I - a_0 - \eta(G(z)))^{-1}, \qquad \Im z>0.
$$

A numerically stable averaged iteration is
$$
b = z(z I - a_0)^{-1},\qquad
G \leftarrow \tfrac12\Big[G + \big(z I - b\,\eta(G)\big)^{-1} b\Big].
$$

## Minimal example

```python
import numpy as np
import free_matrix_laws as fml

n = 3
a0 = np.array([[0,0,0],[0,0,-1],[0,-1,0]], dtype=float)
a1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
a2 = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=float)
A = [a1, a2]

z = 0.1 + 1j*0.02
G, info = fml.solve_cauchy_biased(z, a0, A, tol=1e-10, return_info=True)
print("residual:", info["residual"], "iters:", info["iters"])
```