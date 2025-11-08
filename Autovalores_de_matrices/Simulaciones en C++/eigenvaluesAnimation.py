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
dim = int(parameters[2])
skip = 50

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

# Gráfica y animación de los autovalores

fig, [ax_line, ax_scat] = plt.subplots(ncols=2, figsize=[10, 5])
ax_scat.set_xlim(-1.2 * np.sqrt(dim), 1.2 * np.sqrt(dim))
ax_scat.set_ylim(-1.2 * np.sqrt(dim), 1.2 * np.sqrt(dim))
ax_scat.grid(True, alpha=0.3)
ax_scat.set_aspect('equal')


ax_line.set_xscale('log')
ax_line.set_xlim([10e-2, 10e4])
ax_line.set_ylim([-dim / 2, dim / 2])
ax_line.set_ylabel("Re[eig(W)]")
ax_line.set_xlabel("t")

lines = [ax_line.plot([], [], 'b-', markersize=0.5)[0] for i in range(dim)]

points = ax_scat.scatter(x=[], y=[], c='blue')
circle = Circle(xy=[0, 0], radius=np.sqrt(dim), fill=False)
ax_scat.add_patch(circle)

fig.suptitle(fr"Evolución de los autovalores: dim = {dim}, $\alpha$ = {alpha}, iteraciones = {np.size(data[:, 0])}")
ax_scat.set_title("Dinámica de los autovalores en el plano complejo")
ax_line.set_title("Evolución de la parte real de los autovalores")

def update(frame):
    curr_frame = frame * skip
    points.set_offsets(eigenvalues[curr_frame])
    [lines[i].set_data(np.arange(curr_frame), new_real[:curr_frame, i]) for i in range(dim)]

    return points, *lines

animation = FuncAnimation(fig=fig, func=update, frames=len(data)//skip, blit=True, interval=30)

animation.save("../Animaciones y figuras/Evolucion.mp4")