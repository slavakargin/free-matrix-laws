# free-matrix-laws

Utilities for distributions of **matrix-valued non-commutative variables**:
Cauchy/resolvent transforms, operator-valued maps (e.g., \(\eta(B)=\sum_i A_i B A_i\)),
and small fixed-point solvers.

> Work-in-progress. Feedback welcome.

---

## Install

**From GitHub (read-only users)**
```bash
pip install -U "git+https://github.com/slavakargin/free-matrix-laws.git@main"
```


# Iterating the operator-valued Cauchy transform

We solve the operator-valued semicircle equation
$$
z\,G \;=\; I \;+\; \eta(G)\,G, \qquad \Im z>0,
$$
where $\eta(B)=\sum_{i=1}^s A_i\,B\,A_i^\ast$ is a completely positive (Kraus) map.

## Fixed-point maps

The simplest iteration is
$$
G \;\mapsto\; (\,zI - \eta(G)\,)^{-1}.
$$

Following Helton–Rashidi Far–Speicher (IMRN 2007), a numerically friendlier choice is the *half-averaged* step
$$
G \;\mapsto\; \tfrac12\Big[\,G \;+\; (\,zI - \eta(G)\,)^{-1}\Big],
$$
which damps oscillations while preserving the correct fixed point.

> **Reference.** J. W. Helton, R. Rashidi Far, R. Speicher,  
> *Operator-valued Semicircular Elements: Solving a Quadratic Matrix Equation with Positivity Constraints*, IMRN (2007).


**In code:** use `free_matrix_laws.solve_cauchy_semicircle(z, A, ...)`, which implements the half-averaged step.