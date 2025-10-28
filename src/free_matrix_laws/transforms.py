"""
Numerical transforms for matrix-/operator-valued free probability.
This file was auto-extracted from the notebook `10_semicircle_methods.ipynb`.
Refactor as needed.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple

# ---- Extracted functions from the notebook ----
#(1) covariance map
def eta(B, AA):
  '''
  AA is a list or a tuple of Hermitian matrices A_1, \ldots, A_k
  B is sent to A_1 B A_1 + \ldots + A_k B A_k
  '''
  n = B.shape[0]
  s = len(AA)
  result = np.zeros((n, n), dtype = np.float32)
  for i in range(s):
    result = result + AA[i] @ B @ AA[i]
  return result

#(2a) Iteration step needed to calculate the cauchy transform for
# a matrix semicircle r.v.
def hfs_map(G, z, AA):
  ''' G is a matrix, z is a complex number with positive imaginary part,
  G is mapped to a smoothed version of $(z - \eta(G))^{-1}$. AA is a list of matrices
  needed to define the function $\eta$.
  '''
  n = G.shape[0]
  #return la.inv(z * np.eye(n) - eta(G, AA))
  return (G + la.inv(z * np.eye(n) - eta(G, AA)))/2

#(3a) A function that calculates the Cauchy transform and the distribution
#density for a matrix semicircle r.v.
def get_density(x, AA, eps=0.01, max_iter=10000):
  ''' Calculate the density at the real point x, given the data
  in the tuple of matrices $AA = (A1, \ldots, As)$
  Uses eps as the distance of the point $x + i eps$ from the
  real axis.
  '''
  z = x + 1j * eps
  n = AA[0].shape[0]
  G = 1/z * np.eye(n) #initialization
  diffs = np.zeros((max_iter, 1))
  for i in range(max_iter):
    G1 = hfs_map(G, z, AA)
    diffs[i] = la.norm(G1 - G)
    if la.norm(G1 - G) < 1e-10:
      break
    G = G1
    if i == max_iter - 1:
      print("Warning: no convegence after ", max_iter, "iterations")
  f = (-1/np.pi) * np.imag(np.trace(G)/n)
  #plt.plot(diffs) #this is for diagnostic purposes
  #plt.yscale("log")
  return f


#(2b) the main iteration steps in the calculation of the density for a
#biased matrix semicircle
def hfsb_map(G, z, a, AA):
  ''' G is a matrix, z is a complex number with positive imaginary part,
  a is a bias matrix, AA is a list of matrices
  needed to define the function $\eta$.
  '''
  n = G.shape[0]
  b = z * la.inv(z * np.eye(n) - a)
  W = la.inv(z * np.eye(n) - b @ eta(G, AA)) @ b
  return (G + W)/2
  #return W

#(3b) calculates the Cauchy transform and the density of a biased matrix semicircle
def get_density_B(x, a, AA, eps=0.01, max_iter=10000):
  ''' Calculate the density at the real point x, given the data
  in the tuple of matrices $AA = (A1, \ldots, As)$, and the bias matrix a.
  Uses eps as the distance of the point $x + i eps$ from the
  real axis.
  '''
  z = x + 1j * eps
  n = AA[0].shape[0]
  G = 1/z * np.eye(n) #initialization
  diffs = np.zeros((max_iter, 1))
  for i in range(max_iter):
    G1 = hfsb_map(G, z, a, AA)
    diffs[i] = la.norm(G1 - G)
    if la.norm(G1 - G) < 1e-14:
      break
    G = G1
    if i == max_iter - 1:
      print("Warning: no convegence after ", max_iter, "iterations")
  f = (-1/np.pi) * np.imag(np.trace(G)/n)
  #plt.plot(diffs) #this is for diagnostic purposes
  #plt.yscale("log")
  return f


#(2c) An iteration step in the calculation of the Cauchy transform
# and the density for a polynomial in semicircle r.v.s.
def hfsc_map(G, z, a, AA):
  ''' G is a matrix, z is a complex number with positive imaginary part,
  a is a bias matrix, AA is a list of matrices
  needed to define the function $\eta$.
  '''
  n = G.shape[0]
  b = z * la.inv(Lambda(z, n) - a)
  W = la.inv(z * np.eye(n) - b @ eta(G, AA)) @ b
  return (G + W)/2
  #return W

#(3c) A function that computes the Cauchy transform and the density for
#a polynomial in semicircle r.v.s.
def get_density_C(x, a, AA, eps=0.01, max_iter=10000):
  ''' Calculate the density at the real point x, given the data
  in the tuple of matrices $AA = (A1, \ldots, As)$, and the bias matrix a.
  Uses eps as the distance of the point $x + i eps$ from the
  real axis.
  '''
  z = x + 1j * eps
  n = AA[0].shape[0]
  G = 1/z * np.eye(n) #initialization
  diffs = np.zeros((max_iter, 1))
  for i in range(max_iter):
    G1 = hfsc_map(G, z, a, AA)
    diffs[i] = la.norm(G1 - G)
    if la.norm(G1 - G) < 1e-12:
      break
    G = G1
    if i == max_iter - 1:
      print("Warning: no convegence after ", max_iter, "iterations")
  f = (-1/np.pi) * np.imag(G[0, 0])
  #plt.plot(diffs) #this is for diagnostic purposes
  #plt.yscale("log")
  return f

#(4) creates a random Hermitian Gaussian matrix with approximately semicircle
# distribution.
def random_semicircle(size):
  '''generate a random Hermitian Gaussian matrix of size n-by-n normalized by 1/sqrt(n),
  where n = size'''
  random_matrix = np.random.randn(size, size)
  return (random_matrix + random_matrix.T)/( np.sqrt(2 * size))

#Example of usage:
size = 200
A = random_semicircle(size)
print(la.norm(A - A.T))
e = la.eigvalsh(A)
plt.plot(e)
np.trace(A * A)

def Lambda(z, size, eps = 1E-6):
  ''' Lambda_eps(z) needed to calculate the distribution of a polynomial
  of free random variables.'''
  A = eps * 1.j * np.eye(size)
  A[0, 0] = z
  return A

print(Lambda(1.j, 3))

def G_semicircle(z):
    """
    Computes the Cauchy transform of the semicircle distribution for a given complex number z,
    ensuring that if z has a positive imaginary part, the output has a negative imaginary part,
    and vice versa.

    Parameters:
        z (complex or array-like): The point(s) at which to evaluate the Cauchy transform.

    Returns:
        complex or ndarray: The value(s) of the Cauchy transform at z.
    """
    z = np.asarray(z, dtype=np.complex128)  # Ensure input is treated as complex

    # Compute the discriminant
    discriminant = np.sqrt(z**2 - 4)

    # Ensure the output's imaginary part has the desired symmetry
    discriminant = np.where(discriminant.imag * z.imag < 0, -discriminant, discriminant)

    # Compute the Cauchy transform
    G = (z - discriminant) / 2

    return G

# Example usage
z = 3 + 4j  # Example input
result = G_semicircle(z)
print(f"Cauchy transform at {z} is {result}")

# Test with an array of values
z_array = [3 + 4j, 1 + 1j, 3 - 4j, 1 - 1j]
result_array = G_semicircle(z_array)
print(f"Cauchy transform for array {z_array} is {result_array}")

plt.figure()
z = np.linspace(-4,4) - 0.01j
plt.plot(np.imag(G_semicircle(z)))
z = np.linspace(-4,4) + 0.01j
plt.plot(np.imag(G_semicircle(z)))


def G_matrix_semicircle(w, B, rank):
  ''' computes G(w) for the semicirle B \otimes x,
  rank is the rank of matrix B '''
  w = np.asarray(w, dtype=np.complex128)  # Ensure input is treated as complex
  n = B.shape[0]
  U1, d, U2t = la.svd(B)
  U2 = np.conj(U2t.T)
  #print("U1 =", U1)
  #print(d)
  #print("U2 =", U2)
  #print("should be D: ", np.conj(U1.T) @ B @ U2) #
  A_transf = np.conj(U1.T) @ w @ U2
  #print("A_transf = ", A_transf)

  A11 = A_transf[0:rank, 0:rank]
  A12 = A_transf[0: rank, rank:n]
  A21 = A_transf[rank:n, 0: rank]
  A22 = A_transf[rank:n, rank:n]
  D = np.diag(d[0:rank])
  #print("D = ", D)
  S = A11 - A12 @ la.inv(A22) @ A21
  #print('S = ', S)
  mu, V = la.eig(la.inv(D) @ S)
  #print('mu =', mu)
  #print(V)
  #print('S = ', V @ np.diag(mu) @ la.inv(V))
  #print("G(mu) = ", G_semicircle(mu))
  M11 = V @ np.diag(G_semicircle(mu)) @ la.inv(V) @ la.inv(D)
  #print('M11 = ', M11)
  M = np.block([[M11, np.zeros((rank, n - rank))], [np.zeros((n - rank, rank)), la.inv(A22)]])
  #print("M = ", M)
  G = U2 @ (np.block([[np.eye(rank), np.zeros((rank, n - rank))], [-  la.inv(A22) @ A21 , np.eye(n - rank)]])
       @ M  @ np.block([[np.eye(rank), -A12 @ la.inv(A22)], [np.zeros((n - rank, rank)), np.eye(n - rank)]]))  @ np.conj(U1.T)
  return(G)

def H_matrix_semicircle(w, B, rank):
  ''' This is the h function: h = G(w)^{-1} - w$ '''
  return(la.inv(G_matrix_semicircle(w, B, rank)) - w)


n = 3
rank = 2
A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
z = (0.0 + 1j)
w = z * np.eye(n)

G = G_matrix_semicircle(w, A1, rank)
print("G_matrix_semicircle = ", G)

H = H_matrix_semicircle(w, A1, rank)
print("H_matrix_semicircle = ", H)

def omega(b, AA, rank, max_iter = 10000):
  ''' This computes subordination function for the sum of two semicircle variables.
  AA = (A1, A2), rank is a (rank1, rank2), where rank1 is the rank of matrix A1,
  and rank2 is the rank of matrix A2.
  '''
  W0 = 1.j * np.eye(n) #(initialization)
  A1 = AA[0]
  A2 = AA[1]
  for i in range(max_iter):
    W1 = H_matrix_semicircle(W0, A1, rank = rank[0]) + b
    W2 = H_matrix_semicircle(W1, A2, rank = rank[1]) + b
    if la.norm(W2 - W0) < 1e-12:
      break
    W0 = W2
    if i == max_iter - 1:
      print("Warning: no convergence after ", max_iter, "iterations")
  return W0

#an example
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A0)
print(A1)
print(A2)
AA = (A1, A2)
n = A0.shape[0]

z = .5 + .01j
B = Lambda(z, n) - A0
print(B)

print('result = ', omega(B, AA, rank = (2, 2)))

#(10) Cauchy transform of free Poisson
def G_free_poisson(z, lambda_param):
    """
    Explicit formula for the Cauchy transform of the free Poisson distribution
    with parameter λ.

    Args:
        z (complex): The point at which to evaluate the Cauchy transform.
        lambda_param (float): The parameter λ of the free Poisson law.

    Returns:
        G (complex): The value of the Cauchy transform G(z).
    """

    z = np.asarray(z, dtype=np.complex128)  # Ensure input is treated as complex
    # Compute the interval [a, b] of the support
    a = (1 - np.sqrt(lambda_param))**2
    #print(a)
    b = (1 + np.sqrt(lambda_param))**2
    #print(b)

    # Compute the square root term with correct branch
    sqrt_term = np.sqrt((z - a) * (z - b))
    #sqrt_term = np.sqrt((1 + z - lambda_param)**2 - 4 * z) #alternative expression

    sqrt_term = np.where(sqrt_term.imag * z.imag < 0, -sqrt_term, sqrt_term)

    # Explicit formula for the Cauchy transform
    G = (1 + z - lambda_param - sqrt_term) / (2 * z)
    if lambda_param < 1: #in this case G also has an atom at 0 with weight (1 - lambda)
      G = G + (1 - lambda_param)/z

    return G



lambda_param = 4  # Set λ (parameter of the free Poisson law)
z = 3 + 1j          # Complex number at which to evaluate G(z)

# Compute the Cauchy transform
G_z = G_free_poisson(z, lambda_param)
print(f"Cauchy transform G({z}) for λ={lambda_param}: {G_z}")

#visualization

plt.figure()
#z = np.linspace(-4,4) - 0.01j
#plt.plot(np.imag(G_free_poisson(z, 2)))
m = 100
al = (1 - np.sqrt(lambda_param))**2
au = (1 + np.sqrt(lambda_param))**2
x =  np.linspace(al,au, m)
z = x - 0.01j
plt.plot(x, np.imag(G_free_poisson(z, lambda_param)))
z = x + 0.01j
plt.plot(x, np.imag(G_free_poisson(z, lambda_param)))
plt.grid()

#let's check that this corresponds to a valid density function.
f = - np.imag(G_free_poisson(z, lambda_param))/np.pi
print(sum(f)* (au - al)/m)


# (11) Matrix version of the Cauchy transform for the free Poisson random variable.
def G_matrix_fpoisson(w, B, rank, lambda_param):
  ''' computes G(w) for the free Poisson r.v. B \otimes x,
  rank is the rank of matrix B '''
  w = np.asarray(w, dtype=np.complex128)  # Ensure input is treated as complex
  n = B.shape[0]
  U1, d, U2t = la.svd(B)
  U2 = np.conj(U2t.T)
  A_transf = np.conj(U1.T) @ w @ U2

  A11 = A_transf[0:rank, 0:rank]
  A12 = A_transf[0: rank, rank:n]
  A21 = A_transf[rank:n, 0: rank]
  A22 = A_transf[rank:n, rank:n]
  D = np.diag(d[0:rank])
  S = A11 - A12 @ la.inv(A22) @ A21
  mu, V = la.eig(la.inv(D) @ S)
  M11 = V @ np.diag(G_free_poisson(mu, lambda_param)) @ la.inv(V) @ la.inv(D)
  M = np.block([[M11, np.zeros((rank, n - rank))], [np.zeros((n - rank, rank)), la.inv(A22)]])
  G = U2 @ (np.block([[np.eye(rank), np.zeros((rank, n - rank))], [-  la.inv(A22) @ A21 , np.eye(n - rank)]])
       @ M  @ np.block([[np.eye(rank), -A12 @ la.inv(A22)], [np.zeros((n - rank, rank)), np.eye(n - rank)]]))  @ np.conj(U1.T)
  return(G)

def H_matrix_fpoisson(w, B, rank, lambda_param):
  ''' This is the h function: h = G(w)^{-1} - w$ '''
  return(la.inv(G_matrix_fpoisson(w, B, rank, lambda_param)) - w)


#(13) subordination function for the sum of two matrix random variables.
def omega_sub(b, AA, rank, H1_name="H_matrix_semicircle", H2_name="H_matrix_semicircle",
              H1_kwargs=None, H2_kwargs=None, max_iter=10000):
    '''
    Computes subordination function omega_1(b) for the sum of two free random variables variables.

    AA = (A1, A2), where A1 and A2 are matrices.
    rank = (rank1, rank2), where rank1 is the rank of matrix A1, and rank2 is the rank of matrix A2.
    H1_name, H2_name are string names of the functions to be applied.
    H1_kwargs, H2_kwargs are dictionaries containing additional arguments for H1 and H2.
    '''
    n = AA[0].shape[0]  # Assuming A1 and A2 are square matrices of the same size
    W0 = 1.j * np.eye(n)  # Initialization
    A1, A2 = AA

    # Get function references from globals()
    H1 = globals()[H1_name]
    H2 = globals()[H2_name]

    # Initialize kwargs dictionaries if None
    if H1_kwargs is None:
        H1_kwargs = {}
    if H2_kwargs is None:
        H2_kwargs = {}

    for i in range(max_iter):
        W1 = H1(W0, A1, rank=rank[0], **H1_kwargs) + b
        W2 = H2(W1, A2, rank=rank[1], **H2_kwargs) + b

        if la.norm(W2 - W0) < 1e-12:
            break
        W0 = W2

        if i == max_iter - 1:
            print("Warning: no convergence after", max_iter, "iterations")

    return W0

#(14) generator of a free Poisson matrix
def random_fpoisson(size, lam):
  '''generate a random Hermitian matrix of size n-by-n, where n = size, that have the free Poisson
  distribution with parameter lambda.
  '''
  random_matrix = np.random.randn(size, int(np.floor(size * lam)))
  return (random_matrix @ random_matrix.T) /size

#Example of usage:
size = 200
lam = 4
A = random_fpoisson(size, lam)
print(la.norm(A - A.T))
e = la.eigvalsh(A)
plt.figure()
plt.plot(e)
np.trace(A)/size

#(15) random orthogonal matrix
def random_orthogonal(n):
    # Step 1: Generate a random n x n matrix A
    A = np.random.randn(n, n)

    # Step 2: Perform QR decomposition on A
    Q, R = np.linalg.qr(A)

    # Q is the orthogonal matrix we want
    return Q

# Example usage
n = 3  # Dimension of the matrix
Q = random_orthogonal(n)
print("Random Orthogonal Matrix Q:\n", Q)

# (16) This is a function that calculates the matrix Cauchy transform, provided that
#the scalar Cauchy transform is known.
def G_matrix_custom(w, B, rank, G_name="G_semicircle", G_kwargs=None):
  ''' computes G(w) for the r.v. B \otimes x, where x has a custom measure mu_x above,
  with the scalar Cauchy transform function $G_name$, and
  rank is the rank of matrix B '''

  # Get function references from globals()
  G = globals()[G_name]

  # Initialize kwargs dictionaries if None
  if G_kwargs is None:
    G_kwargs = {}


  w = np.asarray(w, dtype=np.complex128)  # Ensure input is treated as complex
  n = B.shape[0]
  U1, d, U2t = la.svd(B)
  U2 = np.conj(U2t.T)
  A_transf = np.conj(U1.T) @ w @ U2

  A11 = A_transf[0:rank, 0:rank]
  A12 = A_transf[0: rank, rank:n]
  A21 = A_transf[rank:n, 0: rank]
  A22 = A_transf[rank:n, rank:n]
  D = np.diag(d[0:rank])
  S = A11 - A12 @ la.inv(A22) @ A21
  mu, V = la.eig(la.inv(D) @ S)
  M11 = V @ np.diag(G(mu, **G_kwargs)) @ la.inv(V) @ la.inv(D)
  M = np.block([[M11, np.zeros((rank, n - rank))], [np.zeros((n - rank, rank)), la.inv(A22)]])
  G = U2 @ (np.block([[np.eye(rank), np.zeros((rank, n - rank))], [-  la.inv(A22) @ A21 , np.eye(n - rank)]])
       @ M  @ np.block([[np.eye(rank), -A12 @ la.inv(A22)], [np.zeros((n - rank, rank)), np.eye(n - rank)]]))  @ np.conj(U1.T)
  return(G)

# (17) The H-function that corresponds to G_matrix_custom
def H_matrix_custom(w, B, rank, G_name="G_semicircle", G_kwargs=None):
  ''' This is the h function: h = G(w)^{-1} - w$ '''
  return(la.inv(G_matrix_custom(w, B, rank, G_name, G_kwargs)) - w)

n = 3
rank = 2
A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
z = (0.0 + 1j)
w = z * np.eye(n)

G = G_matrix_custom(w, A1, rank, G_name="G_semicircle")
print("G_matrix_custom = ", G)

H = H_matrix_custom(w, A1, rank,  G_name="G_semicircle")
print("H_matrix_custom = ", H)


#(18) The scalar Cauchy transform of an arbitrary discrete distribution
def cauchy_transform_discrete(z, points, weights):
    """
    Computes the Cauchy transform G_mu(z) for a measure defined by
    discrete points and their corresponding weights.

    Parameters:
    z : complex or array-like
        Evaluation point(s) in the complex plane.
    points : list or array-like
        Locations of the discrete measure.
    weights : list or array-like
        Corresponding weights of the measure.
    """
    z = np.asarray(z)[:, np.newaxis]  # Ensure z is a column vector
    points = np.asarray(points)
    weights = np.asarray(weights)
    return np.sum(weights / (z - points), axis=1)

# Example usage
points = np.array([-2, -1, 1])  # Support points
weights = np.array([2/4, 1/4, 1/4])  # Corresponding weights
z_values = np.linspace(-3, 3, 500) + 0.1j  # Evaluate on the upper half-plane
G_values = cauchy_transform_discrete(z_values, points, weights)

# Plot real and imaginary parts
plt.figure(figsize=(8, 5))
plt.plot(z_values.real, G_values.real, label="Re(G_mu(z))", linestyle='dashed')
plt.plot(z_values.real, G_values.imag, label="Im(G_mu(z))")
plt.xlabel("Re(z)")
plt.ylabel("G_mu(z)")
plt.legend()
plt.title("Cauchy Transform of Given Distribution")
plt.grid()
plt.show()

A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
A_test = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
print(A0)
print(A1)
print(A2)
print(A_test)
AA = (A1, A2)
n = A0.shape[0]
w = (0.5 + 0.1j) * np.eye(n)
z = 0.01j
print(la.inv(w - z * A_test))

# Define the matrix-valued function
def matrix_function(z, w, b):
    """Matrix-valued function of a complex variable z."""
    return -(1/np.pi) * la.inv(w - z * b) * np.imag(G_semicircle(z))

print("test:", matrix_function(z, w, A_test))

# Define the real path and shift it slightly above the real line
epsilon = 1e-3  # Small imaginary shift
def shifted_path(x):
    return x + 1j * epsilon

# Integration limits
al, au = -4, 4  # Example real integration limits

# Perform the integration for each matrix entry
def cauchy_matrix_semicircle_0(w, b):
    w = np.asarray(w, dtype=np.complex128)  # Ensure input is treated as complex
    matrix_size = matrix_function(0, w, b).shape  # Get the shape of the matrix
    result = np.zeros(matrix_size, dtype=complex)  # Initialize result matrix
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            # Define the scalar function for the (i, j)-th entry
            def scalar_function(x, w, b):
                z = shifted_path(x)
                return matrix_function(z, w, b)[i, j]

            scalar_func_with_params = partial(scalar_function, w = w, b = b)
            # Perform the numerical integration
            integral_real, _ = quad(lambda x: scalar_func_with_params(x).real, al, au)
            integral_imag, _ = quad(lambda x: scalar_func_with_params(x).imag, al, au)
            result[i, j] = integral_real + 1j * integral_imag  # Save the result
    return result

def H_matrix_semicircle_0(w, A1, eps = 1E-8):
  ''' This is the h function: h = G(w)^{-1} - w$ '''
  return(la.inv(cauchy_matrix_semicircle_0(w, A1)) - w)


# Compute the cauchy transform via integral
result_matrix = cauchy_matrix_semicircle_0(w, A1)
print("Resultant matrix G after integration:")
print(result_matrix)

# Compute the h-function
result_matrix = H_matrix_semicircle_0(w, A1)
print("Resultant matrix H after integration:")
print(result_matrix)



#Let us visualize the result:

m = 10
x = np.linspace(-2, 2, m)
GG = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = cauchy_matrix_semicircle_0((x[i]+ 0.1j) * np.eye(n), A1)
  GG[i] = result[0, 0]
plt.plot(np.imag(GG))
print(GG)

def cauchy_matrix_semicircle_1(w, A1, eps = 1E-8):
  ''' This is function that computes the Cauchy transform of $A1 \otimes X$, where
  X is the standard semicirlce and A1 is an $n\times n$ matrix. The argument is w,
  so we calculate E(w - A1 \otimes X)^{-1}. The parameter eps is for regularization to handle
  the case when A1 is not inverible.'''
  n = A1.shape[0]
  mu, V = la.eig(la.inv(A1 + 1.j * eps * np.eye(n)) @ w)
  return V @ np.diag(G_semicircle(mu)) @ la.inv(V) @ la.inv(A1 + 1.j * eps * np.eye(n))

def H_matrix_semicircle_1(w, A1, eps = 1E-8):
  ''' This is the h function: h = G(w)^{-1} - w$ '''
  return(la.inv(Cauchy_matrix_semicircle_1(w, A1, eps)) - w)


n = 2
A1 = np.eye(n)
A1 = np.array([[0, 1], [1, 0]])

z = (0.0 + 0.01j)
w = z * np.eye(n)

#np.set_printoptions(precision=2)
arr = Cauchy_matrix_semicircle_1(w, A1)
print(arr)
H_matrix_semicircle_1(w, A1)

def Lambda(z, size, eps = 1E-6):
  A = eps * 1.j * np.eye(size)
  A[0, 0] = z
  return A

A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A0)
print(A1)
print(A2)
AA = (A1, A2)
n = A0.shape[0]

z = .5 + .01j
B = Lambda(z, n) - A0
print(B)


W0 = 1.j * np.eye(n) #(initialization)
  #print(W0)
A1 = AA[0]
A2 = AA[1]
  #W1 = H_matrix_semicircle(W0, A1, rank = 2) + B
  #W2 = H_matrix_semicircle(W1, A2, rank = 2) + B
  #print(W2)
max_iter = 40
diffs = np.zeros((max_iter, 1))
for i in trange(max_iter):
  W1 = H_matrix_semicircle(W0, A1, rank = 2) + B
  #print("W1 = ", W1)
  W2 = H_matrix_semicircle(W1, A2, rank = 2) + B
  #print("W2 = ", W2)
  diffs[i] = la.norm(W2 - W0)
  W0 = W2
plt.plot(diffs)
plt.yscale("log")
plt.title("Convergence of the method")
print(W0)

def get_density_anticommutator(x, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  Gxy = G_matrix_semicircle(omega(B, AA, rank = (2, 2)), A1, rank = 2)
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = .5
f = get_density_anticommutator(x)
print(f)

#visualization
m = 50
a = 4
XX = np.linspace(-a, a, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator(x)

print(sum(f)*2*a/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

def get_density_anticommutator_fpoisson(x, lambda_param, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  om = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_fpoisson",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={"lambda_param":lambda_param},
                   H2_kwargs={"lambda_param":lambda_param})
  Gxy = G_matrix_fpoisson(om, A1, rank = 2, lambda_param = lambda_param)
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = 4
f = get_density_anticommutator_fpoisson(x, lambda_param = 4)
print(f)

#visualization

m = 50
al = 2
au = 105
lambda_param = 4
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_fpoisson(x,lambda_param)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)


expect = sum(XX * f)*(au - al)/m
print("expectation = ", expect)

#g = - G_free_poisson(XX + .01j, 100).imag/(np.pi)
#plt.plot(XX, g)

def get_density_anticommutator_S_FP(x, lambda_param, eps = 0.01):
  '''calculates the density of the anticommutator of a semicircle and
  a free poisson r.v. with parameter lambda_param'''
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  om = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_semicircle",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={},
                   H2_kwargs={"lambda_param":lambda_param})
  Gxy = G_matrix_semicircle(om, A1, rank = 2)
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = 4
f = get_density_anticommutator_fpoisson(x, lambda_param = 4)
print(f)

#visualization

m = 50
al = -20
au = 20
lambda_param = 4
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_S_FP(x,lambda_param)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

def get_density_anticommutator_deform(x, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 1/2], [1, 0, 0], [1/2, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  Gxy = G_matrix_semicircle(omega(B, AA, rank = (2, 2)), A1, rank = 2)
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = .5
f = get_density_anticommutator_deform(x)
print(f)

#visualization
m = 50
al = -3
au = 7
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_deform(x)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

def get_density_anticommutator_deform_SFP(x, lambda_param, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 1/2], [1, 0, 0], [1/2, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  om = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_semicircle",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={},
                   H2_kwargs={"lambda_param":lambda_param})
  Gxy = G_matrix_semicircle(om, A1, rank = 2)
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = .5
lam = 4
f = get_density_anticommutator_deform_SFP(x, lambda_param = lam)
print(f)

#visualization
m = 100
al = -18
au = 25
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_deform_SFP(x, lam)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

def cauchy_transform_custom(z):
    """
    Computes the Cauchy transform G_mu(z) of the measure
    mu_X = (1/4)(2δ_{-2} + δ_{-1} + δ_{+1})
    """
    return (1/4) * (2 / (z + 2) + 1 / (z + 1) + 1 / (z - 1))

# Example usage
z_values = np.linspace(-3, 3, 500) + 0.1j  # Evaluate on the upper half-plane
G_values = np.array([cauchy_transform_custom(z) for z in z_values])

# Plot real and imaginary parts
plt.figure(figsize=(8, 5))
plt.plot(z_values.real, G_values.real, label="Re(G_mu(z))", linestyle='dashed')
plt.plot(z_values.real, G_values.imag, label="Im(G_mu(z))")
plt.xlabel("Re(z)")
plt.ylabel("G_mu(z)")
plt.legend()
plt.title("Cauchy Transform of Given Distribution")
plt.grid()
plt.show()

def get_density_anticommutator_deform_custom(x, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 1/2], [1, 0, 0], [1/2, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  om = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_custom",
                   H2_name = "H_matrix_semicircle",
                   H1_kwargs={"G_name":"cauchy_transform_custom"},
                   H2_kwargs={})
  Gxy = G_matrix_custom(om, A1, rank = 2, G_name="cauchy_transform_custom")
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = .5
f = get_density_anticommutator_deform_custom(x)
print(f)

#visualization
m = 100
al = -4
au = 11
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_deform_custom(x)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

def get_density_anticommutator_deform_custom2(x, eps = 0.01):
  A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
  A1 = np.array([[0, 1, 1/2], [1, 0, 0], [1/2, 0, 0]])
  A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
  AA = (A1, A2)
  n = A0.shape[0]
  z = x + eps * 1j
  B = Lambda(z, n) - A0
  om = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_custom",
                   H2_name = "H_matrix_custom",
                   H1_kwargs={"G_name":"cauchy_transform_discrete",
                              "G_kwargs":{"points":np.array([-2, -1, 1]), "weights": np.array([2/4, 1/4, 1/4])}},
                   H2_kwargs={"G_name":"cauchy_transform_discrete",
                              "G_kwargs":{"points":np.array([1, 3]), "weights": np.array([1/2, 1/2])}})
  Gxy = G_matrix_custom(om, A1, rank = 2, G_name="cauchy_transform_discrete",
                        G_kwargs={"points":np.array([-2, -1, 1]), "weights": np.array([2/4, 1/4, 1/4])})
  f = (-1/np.pi) * Gxy[0,0].imag
  return f

x = .5
f = get_density_anticommutator_deform_custom2(x)
print(f)

#visualization
m = 100
al = -10
au = 10
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_anticommutator_deform_custom2(x)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

