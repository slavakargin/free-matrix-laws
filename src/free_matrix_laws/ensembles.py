import numpy as np

def random_semicircle(n: int,
                      field: str = "real",
                      variance: float = 1.0,
                      seed=None):
    r'''
    Generate an $n\times n$ Hermitian (self-adjoint) Wigner matrix whose eigenvalues
    follow (as $n\to\infty$) a semicircle law with variance parameter $c=\texttt{variance}$,
    i.e. support $[-2\sqrt{c},\,2\sqrt{c}]$.

    Conventions (entry variances):
    - $\texttt{field}=\text{"real"}$ (GOE normalization):
      off-diagonal $\operatorname{Var}(H_{ij}) = c/n$, diagonal $\operatorname{Var}(H_{ii}) = 2c/n$.
    - $\texttt{field}=\text{"complex"}$ (GUE normalization):
      off-diagonal $\mathbb E|H_{ij}|^2 = c/n$, diagonal $\operatorname{Var}(H_{ii}) = c/n$.

    Parameters
    ----------
    n : int
        Matrix size.
    field : {"real","complex"}, default "real"
        Real symmetric (GOE-type) or complex Hermitian (GUE-type).
    variance : float, default 1.0
        Semicircle variance parameter $c>0$. The spectrum concentrates on
        $[-2\sqrt{c},\,2\sqrt{c}]$ in the large-$n$ limit.
    seed : int or numpy.random.Generator, optional
        Random seed or generator.

    Returns
    -------
    H : ndarray
        Hermitian matrix (dtype float64 for real, complex128 for complex).

    Notes
    -----
    Implementation uses the convenient symmetrize-and-scale recipe:

    - Real: $H = \dfrac{X + X^\top}{\sqrt{2n}}$, $X_{ij}\sim\mathcal N(0,1)$.
      This yields $\operatorname{Var}(H_{ij})=1/n$ (off-diagonal), $\operatorname{Var}(H_{ii})=2/n$.
    - Complex: draw $Z_{ij}\sim \mathcal{CN}(0,1)$ (i.e. $(X+iY)/\sqrt{2}$ with $X,Y\sim\mathcal N(0,1)$),
      then $H = \dfrac{Z + Z^\ast}{\sqrt{2n}}$, giving $\mathbb E|H_{ij}|^2=1/n$ and
      $\operatorname{Var}(H_{ii})=1/n$. Finally multiply by $\sqrt{c}$.
    '''
    if variance <= 0:
        raise ValueError("variance must be > 0")
    rng = np.random.default_rng(seed)

    if field == "real":
        X = rng.standard_normal((n, n))
        H = (X + X.T) / np.sqrt(2.0 * n)          # GOE normalization
        if variance != 1.0:
            H = np.sqrt(variance) * H
        return H

    if field == "complex":
        X = rng.standard_normal((n, n))
        Y = rng.standard_normal((n, n))
        Z = (X + 1j * Y) / np.sqrt(2.0)           # CN(0,1) entries
        H = (Z + Z.conj().T) / np.sqrt(2.0 * n)   # GUE normalization
        if variance != 1.0:
            H = np.sqrt(variance) * H
        return H

    raise ValueError('field must be "real" or "complex"')
