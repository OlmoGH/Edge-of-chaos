import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import pandas as pd


# Datos de los autovalores y de los parámetros
with open("Eigenvalues.txt", "r") as f:
    f.readline().split()
    parameters = [float(x) for x in f.readline().strip().split(",")]

alpha = parameters[0]
dt = parameters[1]
dim = parameters[2]
skip = 1

data = pd.read_csv("Eigenvalues.txt", sep=',', header=0, skiprows=2).to_numpy()
real = data[:, 0::2]
imag = data[:, 1::2]
eigenvalues = np.stack((real, imag), axis=2)

# Ordenamos los autovalores para que no haya saltos raros
sorted_indices = np.argsort(eigenvalues[:, :, 0], axis=1)
rows = np.arange(eigenvalues.shape[0])[:, np.newaxis]
cols = sorted_indices
eigenvalues = eigenvalues[rows, cols]
new_real = eigenvalues[:, :, 0]
new_imag = eigenvalues[:, :, 1]

print(np.shape(eigenvalues))

# Gráfica de los autovalores
timescale = np.linspace(0, np.shape(data)[0] * dt, np.shape(data)[0])
fig_plot, ax_real = plt.subplots()
ax_real.set_xscale('log')
ax_real.set_xlim([10e-2, 10e4])
ax_real.plot(timescale, new_real, 'b-', markersize=0.5)
plt.show()

# Animación de los autovalores

fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

points = ax.scatter(x=[], y=[], c='blue')
circle = Circle(xy=[0, 0], radius=1, fill=False)
ax.add_patch(circle)

def update(frame):
    frame = frame * skip
    points.set_offsets(eigenvalues[frame])

    return points,

animation = FuncAnimation(fig=fig, func=update, frames=len(data)//skip, blit=True, interval=30)

plt.show()