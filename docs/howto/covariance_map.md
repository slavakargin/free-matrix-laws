# How to use `covariance_map`

Below is a minimal example. If you can't see the code block, your Markdown
extensions/nav paths aren't wired correctly.

```python
print("hello!")
```

    import numpy as np
    from free_matrix_laws import covariance_map as eta

    n, s = 3, 2
    A1 = np.eye(n)
    A2 = 2*np.eye(n)
    B  = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])

    val = eta(B, [A1, A2])          # list of A_i
    # or: val = eta(B, np.stack([A1, A2], axis=0))  # (s,n,n) stack

    # sanity check against a loop
    val_loop = A1 @ B @ A1 + A2 @ B @ A2
    assert np.allclose(val, val_loop)
