"""
Code snippets from the notebook that were not inside a function.
Refactor into library functions and tests as needed.
"""

import numpy as np

# ---- Begin snippets ----

# ---- Snippet 1 ----
import numpy as np
import numpy.linalg as la
from scipy.integrate import quad
from functools import partial
import seaborn as sns

from matplotlib import pyplot as plt, patches
from tqdm.notebook import tqdm, trange

%matplotlib inline
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ---- Snippet 2 ----
B = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
print(B)
A1 = 2.j * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
print(A1)
A2 = 1.j * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
A3 = 1.j * np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]])
AA = (A1, A2, A3)
etaB = eta(B, AA)
etaB

# ---- Snippet 3 ----
z = .01j
n = 3
G0 = .1/z * np.eye(n)
G = hfs_map(G0, z, AA)
print(G)

max_iter = 150
diffs = np.zeros((max_iter, 1))
for i in trange(max_iter):
  G1 = hfs_map(G, z, AA)
  diffs[i] = la.norm(G1 - G)
  G = G1
plt.plot(diffs)
plt.yscale("log")
plt.title("Convergence of the method")
print(G)

# ---- Snippet 4 ----
f = get_density(0., AA)
print(f)

# ---- Snippet 5 ----
a = 8
m = 100
XX = np.linspace(-a, a, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
  f[i] = get_density(x, AA)

print(sum(f)*2*a/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)


# ---- Snippet 6 ----
size = 200
zero_m = np.zeros((size, size))
T = 10
EE = np.zeros((3 * size,T))
for count in range(T):
  S1 = random_semicircle(size)
  S2 = random_semicircle(size)
  S3 = random_semicircle(size)
  S = 1.j * np.block([[zero_m, 2 * S1 + S3, S2], [-2 * S1 - S3, zero_m, -S3], [-S2, S3, zero_m]])
  #la.norm(S - np.conj(S.T))
  e = la.eigvalsh(S)
  EE[:,count] = e

EE = EE.reshape(-1)

#plt.plot(EE)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=20, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()



# ---- Snippet 7 ----
A1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
A2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
A3 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A1)
print(A2)
print(A3)
AA = (A1, A2, A3)
a = 5
XX = np.linspace(-a, a, 100)
m = XX.size
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
  f[i] = get_density(x, AA)

print(sum(f)*2*a/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)

# ---- Snippet 8 ----
size = 200
zero_m = np.zeros((size, size))
T = 30
EE = np.zeros((3 * size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  C = random_semicircle(size)
  S = np.block([[A, B, C], [B, A, B], [C, B, A]])
  #la.norm(S - np.conj(S.T))
  e = la.eigvalsh(S)
  EE[:,count] = e

EE = EE.reshape(-1)

#plt.plot(EE)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()


# ---- Snippet 9 ----
#usage example
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A0)
print(A1)
print(A2)
AA = (A1, A2)

'''
#checking convergence
z = 1 + .01j
n = 3
G0 = 1/z * np.eye(n)
G = hfsb_map(G0, z, A0, AA)
print(G)
max_iter = 1500
diffs = np.zeros((max_iter, 1))
for i in trange(max_iter):
  G1 = hfsb_map(G, z, A0, AA)
  diffs[i] = la.norm(G1 - G)
  G = G1
plt.plot(diffs)
plt.yscale("log")
plt.title("Convergence of the method")
print(G)
'''
f = get_density_B(1., A0, AA)
print(f)
a = 4
m = 100
XX = np.linspace(-a, a, 200)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_B(x, A0, AA)

print(sum(f)*2*a/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

# ---- Snippet 10 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((3 * size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  C = random_semicircle(size)
  S = np.block([[zero_m, B, C], [B, zero_m, -ones_m], [C, -ones_m, zero_m]])
  #print(la.norm(S - np.conj(S.T)))
  e = la.eigvalsh(S)
  EE[:,count] = e

EE = EE.reshape(-1)

#plt.plot(EE)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 11 ----
#usage example
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A0)
print(A1)
print(A2)
AA = (A1, A2)
z = .5 + .01j
n = 3
G0 = 1/z * np.eye(n)
G = hfsc_map(G0, z, A0, AA)
print(G)

max_iter = 200
diffs = np.zeros((max_iter, 1))
for i in trange(max_iter):
  G1 = hfsc_map(G, z, A0, AA)
  diffs[i] = la.norm(G1 - G)
  G = G1
plt.plot(diffs)
plt.yscale("log")
plt.title("Convergence of the method")
print(G)

# ---- Snippet 12 ----
f = get_density_C(1., A0, AA)
print(f)
a = 4
m = 200
XX = np.linspace(-a, a, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_C(x, A0, AA)

print(sum(f)*2*a/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

get_density_C(.5, A0, AA)

# ---- Snippet 13 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  e = la.eigvalsh(A @ B + B @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 14 ----
z = .5 + .01j
n = 3
B = Lambda(z, n) - A0
print(B)

W0 = 1.j * np.eye(n) #(initialization)
print(W0)

W1 = H_matrix_semicircle_0(W0, A1) + B
W2 = H_matrix_semicircle_0(W1, A2) + B
print(W2)

max_iter = 30
diffs = np.zeros((max_iter, 1))
for i in trange(max_iter):
  W1 = H_matrix_semicircle_0(W0, A1) + B
  #print("W1 = ", W1)
  W2 = H_matrix_semicircle_0(W1, A2) + B
  #print("W2 = ", W2)
  diffs[i] = la.norm(W2 - W0)
  W0 = W2
plt.plot(diffs)
plt.yscale("log")
plt.title("Convergence of the method")
print(W0)

# ---- Snippet 15 ----
-cauchy_matrix_semicircle_0(W0, A1)[0,0].imag/np.pi

# ---- Snippet 16 ----
n = 2
rank = 2
A1 = np.eye(n)
A1 = np.array([[0, 1], [1, 0]])

#A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
n = 3
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])





z = (0.0 + 1j)
w = z * np.eye(n)

G = cauchy_matrix_semicircle_0(w, A1)
print(G)
print('old appoach = \n', cauchy_matrix_semicircle_0(w, A1))
print(G_semicircle(z))

#H = H_matrix_semicircle_0(w, A1, rank)
#print(H)

#Let us visualize the result:
m = 10
x = np.linspace(-2, 2, m)
GG = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = cauchy_matrix_semicircle_0((x[i]+ 0.1j) * np.eye(n), A1)
  GG[i] = result[0, 0]
plt.plot(np.imag(GG))
print(GG)

m = 10
x = np.linspace(-2, 2, m)
GG_new = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = cauchy_matrix_semicircle_1((x[i]+ 0.1j) * np.eye(n), A1)
  GG_new[i] = result[0, 0]
plt.plot(np.imag(GG_new) + 0.01)
print(GG_new)

# ---- Snippet 17 ----

A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])



#Let us visualize the result:
m = 10
x = np.linspace(-2, 2, m)
GG = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = cauchy_matrix_semicircle_0((x[i]+ 0.1j) * np.eye(n), A1)
  GG[i] = result[0, 0]
plt.plot(np.imag(GG))
print(GG)

m = 10
x = np.linspace(-2, 2, m)
GG_new = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = G_matrix_semicircle((x[i]+ 0.1j) * np.eye(n), A1, rank = 2)
  GG_new[i] = result[0, 0]
plt.plot(np.imag(GG_new) + 0.01)
print(GG_new)

plt.figure()

m = 10
x = np.linspace(-2, 2, m)
HH = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = H_matrix_semicircle_0((x[i]+ 0.1j) * np.eye(n), A1)
  HH[i] = result[0, 0]
plt.plot(np.imag(HH))
print(HH)

m = 10
x = np.linspace(-2, 2, m)
HH_new = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = H_matrix_semicircle((x[i]+ 0.1j) * np.eye(n), A1, rank = 2)
  HH_new[i] = result[0, 0]
plt.plot(np.imag(HH_new) + 0.01)
print(HH_new)



# ---- Snippet 18 ----
#See the definition of the function in the beginning of the notebook

#let us do some visualization
m = 50
tt = np.linspace(-2, 2, m)
om = np.zeros((m, 1), dtype = np.complex128)
for i in range(m):
  om[i] = omega(Lambda(tt[i] + 0.01j, n) - A0, (AA), rank = (2,2))[0,0]
#print(om)
plt.plot(np.imag(om))
plt.plot(np.real(om))

# ---- Snippet 19 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  e = la.eigvalsh(A @ B + B @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 20 ----
n = 3
lambda_param = 4
A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
rank = 2
z = (0.0 + 1j)
w = z * np.eye(n)

G = G_matrix_fpoisson(w, A1, rank, lambda_param)
print("G_matrix_fpoisson = ", G)


A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
#A1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
m = 100
x = np.linspace(-2, 2, m)
GG_new = np.zeros(m, dtype=np.complex128)
for i in trange(m):
  result = H_matrix_fpoisson((x[i]+ 0.1j) * np.eye(n), A1, rank = 2, lambda_param = 0.5)
  GG_new[i] = result[0, 0]
plt.plot(np.imag(GG_new) + 0.01)
print(GG_new)

# ---- Snippet 21 ----
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

result = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_fpoisson",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={"lambda_param":4},
                   H2_kwargs={"lambda_param":4})

# ---- Snippet 22 ----
#let us do some visualization
m = 50
tt = np.linspace(1, 9, m)
om = np.zeros((m, 1), dtype = np.complex128)
for i in range(m):
  B = Lambda(tt[i] + 0.01j, n) - A0
  om[i] = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_fpoisson",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={"lambda_param":4},
                   H2_kwargs={"lambda_param":4})[0,0]
#print(om)
plt.plot(tt, np.imag(om))
plt.plot(tt, np.real(om))

# ---- Snippet 23 ----
size = 200
lam = 4
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_fpoisson(size, lam)
  B = random_fpoisson(size, lam)
  e = la.eigvalsh(A @ B + B @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 24 ----
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

result = omega_sub(B, (A1, A2), rank = (2,2), H1_name = "H_matrix_semicircle",
                   H2_name = "H_matrix_fpoisson",
                   H1_kwargs={},
                   H2_kwargs={"lambda_param":4})

# ---- Snippet 25 ----
size = 200
lam = 4
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_fpoisson(size, lam)
  e = la.eigvalsh(A @ B + B @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 26 ----
#usage example
A0 = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
A1 = np.array([[0, 1, 1/2], [1, 0, 0], [1/2, 0, 0]])
A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
print(A0)
print(A1)
print(A2)
AA = (A1, A2)
f = get_density_C(1., A0, AA)
print(f)
al = -3
au = 8
m = 200
XX = np.linspace(al, au, m)
f = np.zeros(XX.shape)
for i, x in enumerate(XX):
   f[i] = get_density_C(x, A0, AA)

print(sum(f)*(au - al)/m) #just to check that the integral of the density is 1 (approximately)
plt.plot(XX, f)
plt.grid(True)

# ---- Snippet 27 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  e = la.eigvalsh(A @ B + B @ A + A @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 28 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_semicircle(size)
  e = la.eigvalsh(A @ B + B @ A + A @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 29 ----
size = 200
lam = 4
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  A = random_semicircle(size)
  B = random_fpoisson(size, lam)
  e = la.eigvalsh(A @ B + B @ A + A @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 30 ----
size = 200
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  # Define the values for the diagonal
  diagonal_values = np.concatenate([np.full(100, -2), np.full(50, -1), np.full(50, 1)])
  A = np.diag(diagonal_values)
  B = random_semicircle(size)
  e = la.eigvalsh(A @ B + B @ A + A @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# ---- Snippet 31 ----
size = 400
zero_m = np.zeros((size, size))
ones_m = np.eye(size)
T = 30
EE = np.zeros((size,T))
for count in range(T):
  # Define the values for the diagonal
  diagonal_values = np.concatenate([np.full(200, -2), np.full(100, -1), np.full(100, 1)])
  A = np.diag(diagonal_values)
  diagonal_values = np.concatenate([np.full(200, 1), np.full(200, 3)])
  B = np.diag(diagonal_values)
  Q = random_orthogonal(size)
  B = Q @ B @ Q.T

  e = la.eigvalsh(A @ B + B @ A + A @ A)
  EE[:,count] = e

EE = EE.reshape(-1)

plt.figure()
# Plot histogram with density
plt.hist(EE, bins=30, density=True, edgecolor='black', alpha=0.6, label="Histogram")
# Plot theoretical density curve
plt.plot(XX, f, color='red', label="Theoretical Density")

# Add labels and legend
plt.title("Histogram with Theoretical Density Curve")
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

