# Random Wigner Gaussian matrices with semicircle scaling

This shows how to sample GOE/GUE-type matrices whose eigenvalues follow (as $n\to\infty$) a semicircle law with variance $c$ (support $[-2\sqrt{c},\,2\sqrt{c}]$).

## Quick start

```python
import numpy as np
import free_matrix_laws as fml

n = 300

# Real symmetric (GOE-type), variance c=1
H_goe = fml.random_semicircle(n, field="real", variance=1.0, seed=0)

# Complex Hermitian (GUE-type), variance c=0.5
H_gue = fml.random_semicircle(n, field="complex", variance=0.5, seed=1)

# Hermitian checks
assert np.allclose(H_goe, H_goe.T)
assert np.allclose(H_gue, H_gue.conj().T)
```

### Options

* **field**: `"real"` (GOE) or `"complex"` (GUE).
* **variance**: $c>0$; spectrum concentrates on $[-2\sqrt{c},,2\sqrt{c}]$.
* **seed**: integer or `numpy.random.Generator` for reproducibility.

### Sanity checks

```python
import numpy as np

n, c = 200, 1.7
H = fml.random_semicircle(n, field="real", variance=c, seed=0)

# Off-diagonal variance ≈ c/n (finite-n tolerance)
i, j = np.triu_indices(n, 1)
v_emp = np.var(H[i, j])
print("empirical offdiag var ~", v_emp, "target ~", c/n)

# Spectral radius ≈ 2*sqrt(c) (coarse finite-n check)
smax = np.linalg.svd(H, compute_uv=False)[0]
print("||H|| ~", smax, "target ~", 2*np.sqrt(c))
```


### Histogram

```python
import numpy as np
import matplotlib.pyplot as plt

n, c = 600, 1.0
H = fml.random_semicircle(n, field="real", variance=c, seed=42)
lam = np.linalg.eigvalsh(H)

plt.hist(lam, bins=60, density=True)
plt.xlabel("eigenvalue")
plt.ylabel("density")
plt.title("Empirical spectrum vs. semicircle (c=1)")
plt.show()
```