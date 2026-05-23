import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate(STEPS, W, X_0, DT):
    DIM = X_0.shape[0]
    X_time_series = np.zeros((STEPS, DIM))
    X_time_series[0] = X_0
    for t in range(1, STEPS):
        if t % 1000 == 0:
            print(f"Paso {t}/{STEPS}")
        for i in range(DIM):
            sum = 0
            for j in range(DIM):
                sum += W[i, j] * X_time_series[t-1, j]

            X_time_series[t, i] = X_time_series[t-1, i] + DT * sum
    
    return X_time_series

@njit(fastmath=True)
def hopfield(STEPS, W, X_0, DT):
    DIM = X_0.shape[0]
    X_time_series = np.zeros((STEPS, DIM))
    X_time_series[0] = X_0
    for t in range(1, STEPS):
        if t % 10 == 0:
            print(f"Paso {t}/{STEPS}")
        for i in range(DIM):
            sum = 0
            for j in range(DIM):
                sum += W[i, j] * X_time_series[t-1, j]

            X_time_series[t, i] = np.tanh(sum)
    
    return X_time_series
DIM = 400
STEPS = 100
DT = 0.01
X_0 = np.random.standard_normal(DIM)

# u = np.ones(DIM)
# u[::2] = -1

# u = u / np.linalg.norm(u)

# v = np.ones(DIM) / np.sqrt(DIM)

u = np.loadtxt("Autovalores_de_matrices/Simulaciones en Python/Cara.txt", delimiter=',')
u = np.astype(u*2 - 1, int)
print(u.shape)

v = np.loadtxt("Autovalores_de_matrices/Simulaciones en Python/Fresa.txt", delimiter=',')
v = np.astype(v*2 - 1, int)

W = np.outer(u, v) - np.outer(v, u)

eigenvalues = np.linalg.eigvals(W)

time_series = hopfield(STEPS, W, X_0, DT)

plt.plot(time_series[:, 0])
plt.show()

L = int(np.sqrt(DIM))
np.empty((L, L))

fig, ax = plt.subplots()
img = ax.imshow(np.empty((L, L)), aspect='auto', cmap='gray')
def update(frame):
    bin_image = time_series[frame].reshape((L, L))
    img.set_data(bin_image)

    return img,

animation = FuncAnimation(fig, func=update, frames=STEPS, interval=500, blit=True)
plt.show()