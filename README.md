# free-matrix-laws

Tools for **matrix-/operator-valued free probability**: covariance/Kraus maps, fixed-point solvers for Cauchy (resolvent) transforms, matrix semicircle (unbiased/biased) densities, and small random-matrix helpers.

## Documentation
- **Online docs:** https://slavakargin.github.io/free-matrix-laws/
  - How-to guides and auto-generated API (mkdocstrings).

## Install

### Users (read-only from GitHub)
```bash
pip install -U "git+https://github.com/slavakargin/free-matrix-laws.git@main"
```

### Dev install (editable)
```bash
git clone https://github.com/slavakargin/free-matrix-laws.git
cd free-matrix-laws
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .                                     # editable install
```

## Quickstart

```python
import numpy as np
from free_matrix_laws import (
    covariance_map,            # a.k.a. eta(B, A)
    solve_cauchy_semicircle,   # operator-valued G(z) for matrix semicircle
    semicircle_density,        # scalar density via normalized trace
    biased_semicircle_density, # density for a0 + sum A_i ⊗ X_i
    random_semicircle,         # GOE-like helper (and variants, see docs)
)

# Example Kraus operators (n×n)
n, s = 3, 2
A1 = np.eye(n)
A2 = 2*np.eye(n)
A  = [A1, A2]

# 1) Scalar density of matrix semicircle at x
x  = 0.0
fx = semicircle_density(x, A, eps=1e-2)

# 2) Biased case: S = a0 + sum_i A_i ⊗ X_i
a0 = np.array([[0,0,0],[0,0,-1],[0,-1,0]], dtype=float)
fb = biased_semicircle_density(x, a0, A, eps=1e-2)

# 3) Operator-valued Cauchy transform G(z)
z  = 0.2 + 1j*1e-2
G  = solve_cauchy_semicircle(z, A)

# 4) Covariance map η(B) = sum_i A_i B A_i*
B  = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
etaB = covariance_map(B, A)

# 5) Random GOE-like matrix with semicircle scaling
M = random_semicircle(n)   # see docs for options
```

**Tips**
- For densities at real $x$, use $z=x+i\varepsilon$ with $\varepsilon \approx 10^{-2}$–$10^{-3}$.
- Warm starts (`G0`) from a nearby $z$ can accelerate fixed-point convergence.
- Input shapes: `A` may be a list/tuple of $(n,n)$ arrays or a stacked array of shape $(s,n,n)$.

## Tests
```bash
python -m pytest -q
```

## References
- J.W. Helton, R. Rashidi Far, R. Speicher, *Operator-valued Semicircular Elements: Solving a Quadratic Matrix Equation with Positivity Constraints*, IMRN (2007).
- R. Speicher, *Combinatorial Theory of the Free Product with Amalgamation and Operator-Valued Free Probability Theory*, Memoirs AMS 132 (1998).

## Conventions
Matrices are $n\times n$. Expectations use the normalized trace unless stated.

## License
See `LICENSE`.
