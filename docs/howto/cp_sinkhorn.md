# How to normalize a CP map via symmetric OSI

Given a CP map
$$
T(X)=\sum_{i=1}^s A_i X A_i^\ast,
$$
one symmetric OSI step forms
$$
c_1 = \big(T(I)\big)^{-1/2}, \qquad
c_2 = \big(T^\ast(I)\big)^{-1/2},
$$
and produces scaled Kraus operators
$$
B_i = c_1 A_i c_2,
$$
so that $\mathcal S(T)(X)=\sum_i B_i X B_i^\ast$.
Iterating this step often moves $T$ toward the **doubly stochastic** class ($T(I)=I$, $T^\ast(I)=I$).

## Prerequisites

```python
import numpy as np
from free_matrix_laws import (
    symmetric_sinkhorn_scale,
    symmetric_sinkhorn_apply,
    symmetric_osi,
    ds_distance,
)
```

Your Kraus operators can be a **list/tuple** of shape `(n,n)` arrays or a **stacked** array of shape `(s,n,n)`.

## One step: scale the Kraus operators

```python
n, s = 4, 3
rng = np.random.default_rng(0)
A = rng.standard_normal((s, n, n)) + 1j * rng.standard_normal((s, n, n))

B = symmetric_sinkhorn_scale(A)     # returns scaled Kraus {B_i}
print("DS distance before:", ds_distance(A))
print("DS distance after :", ds_distance(B))
```

Here `ds_distance(T) = ||T(I)-I||_F^2 + ||T^\ast(I)-I||_F^2` is a convenient progress metric.

## Apply the scaled map to a matrix $X$

```python
X = np.eye(n)
Y = symmetric_sinkhorn_apply(X, A)  # uses one symmetric scaling step internally
# Equivalent to:
# B = symmetric_sinkhorn_scale(A, preserve_input_type=False)
# Y2 = sum(Bi @ X @ Bi.conj().T for Bi in B)
```

## Iterate to near doubly stochastic

```python
B_final, info = symmetric_osi(A, maxiter=50, tol=1e-10, return_history=True)
print("iters:", info["iters"])
print("final DS distance:", info["ds"])
```

If `info["ds"]` is tiny (e.g., $\le 10^{-10}$), the map is numerically close to DS.

## Tips

* **Singular cases:** If $T(I)$ or $T^\ast(I)$ is ill-conditioned, the implementation uses a small eigenvalue floor `eps` inside the inverse square roots. Raise `eps` if you see numerical instabilities.
* **Container type:** If you pass a list, you’ll get a list back; for a stacked array, a stacked array comes back (configurable via `preserve_input_type`).
* **Composition:** Returning scaled Kraus operators is convenient when you want to reuse them with other routines (e.g., covariance maps or resolvent solvers).

## See also

* API: [opvalued (Sinkhorn helpers)](../api/opvalued.md)

```
```

