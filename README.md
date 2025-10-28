# free-matrix-laws

Calculations and notebooks on **matrix-/operator-valued free probability**:
Cauchy/resolvent transforms, \(R\) and \(S\) transforms, and empirical experiments.

## Layout
- `notebooks/10_semicircle_methods.ipynb`: your original notebook, preserved.
- `src/free_matrix_laws/transforms.py`: functions auto-extracted from the notebook (or stubs if none were found).
- `src/free_matrix_laws/notebook_scratch.py`: non-function code to refactor.
- `tests/`: smoke tests (expand as you refactor).

## Dev install
```bash
pip install -e .
python -c "import free_matrix_laws as fml; print('OK', dir(fml)[:5])"
```

## Conventions
Matrices are \(n\times n\). Expectations use normalized trace unless stated.
