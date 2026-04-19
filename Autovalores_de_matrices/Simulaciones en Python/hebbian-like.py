import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit(fastmath=True)
def simulate(STEPS, W, X_0, DT):
    DIM = X_0.shape[0]
    X = np.zeros((STEPS, DIM))
    X[0] = X_0
    temp_X = np.empty_like(X_0)
    phi = np.empty_like(X_0)
    k1 = np.empty_like(X_0)
    k2 = np.empty_like(X_0)
    k3 = np.empty_like(X_0)
    k4 = np.empty_like(X_0)

    # Integramos todos los pasos temporales
    for step in range(1, STEPS):

        # Calculamos k1
        for i in range(DIM):
            k1[i] = -X[step-1, i]
            for j in range(DIM):
                k1[i] += W[i, j] * np.tanh(X[step-1, j])

        # Calculamos k2
        for i in range(DIM):
            temp_X[i] = X[step-1, i] + 0.5 * k1[i] * DT
        for i in range(DIM):
            k2[i] = -temp_X[i]
            for j in range(DIM):
                k2[i] += W[i, j] * np.tanh(temp_X[j])

        # Calculamos k3
        for i in range(DIM):
            temp_X[i] = X[step-1, i] + 0.5 * k2[i] * DT
        for i in range(DIM):
            k3[i] = -temp_X[i]
            for j in range(DIM):
                k3[i] += W[i, j] * np.tanh(temp_X[j])

        # Calculamos k4
        for i in range(DIM):
            temp_X[i] = X[step-1, i] + k3[i] * DT
        for i in range(DIM):
            k4[i] = -temp_X[i]
            for j in range(DIM):
                k4[i] += W[i, j] * np.tanh(temp_X[j])

        # Integramos con todos los asos intermedios
        for i in range(DIM):
            X[step, i] = X[step-1, i] + DT * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
        
    
    return X

DIM = 1000
STEPS = 10000
DT = 0.01
X = np.random.standard_normal(DIM)
rng_u = np.random.default_rng(42)
u = rng_u.standard_normal(DIM)
u = u / np.linalg.norm(u)

rng_v = np.random.default_rng(69)
v = rng_v.standard_normal(DIM)
v = v - np.dot(u, v) * u
v = v / np.linalg.norm(v)

W = np.outer(u, v) - np.outer(v, u) + (np.outer(u, u) + np.outer(v, v)) * 4

time_series = simulate(STEPS, W, X, DT)

p_u = np.dot(time_series, u)
p_v = np.dot(time_series, v)

r1 = np.random.standard_normal(DIM)
r1 = r1 / np.linalg.norm(r1)

r2 = np.random.standard_normal(DIM)
r2 = r2 - np.dot(r1, r2) * r1
r2 = r2 / np.linalg.norm(r2)

p_r1 = np.dot(time_series, r1)
p_r2 = np.dot(time_series, r2)

plt.plot(np.sqrt(p_u**2 + p_v**2))
plt.plot(np.sqrt(p_r1**2 + p_r2**2))
plt.show()