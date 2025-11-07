import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# Parámetros del sistema
N = 100
W = np.random.normal(loc=0, scale=1, size=(N, N))
x = np.random.normal(loc=0, scale=1, size=(1, N))
alpha = 0.01
paso = 0.01
fps = 100
t_animacion = 1000

# Funciones de evolución
def evolucionar_x():
    global x, W
    x = x @ (np.identity(N) + paso * W)

def evolucionar_w():
    global x, W
    W = W + alpha * paso * (np.identity(N) - np.outer(x, x))

# Autovalores iniciales
eigenvalues = np.empty((fps * t_animacion, N), dtype=complex)
eigenvalues[0] = np.linalg.eigvals(W)

# Configuración de gráficos
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_aspect('equal')

circulo = Circle((0, 0), radius=np.sqrt(N), fill=False)
ax1.add_patch(circulo)
dispersion = ax1.scatter([], [], s=10)

lines = [ax2.plot([], [], markersize=2)[0] for _ in range(N)]
ax2.set_xscale('log')
ax2.set_ylim([-N, N])

# Función de actualización
def update(frame):
    frames = np.linspace(0, frame * paso, frame + 1)
    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)

    dispersion.set_offsets(np.c_[real[frame], imag[frame]])

    for i in range(N):
        lines[i].set_data(frames, real[:frame + 1, i])

    evolucionar_x()
    evolucionar_w()

    new_eigenvalues = np.linalg.eigvals(W)
    if frame + 1 < len(eigenvalues):
        eigenvalues[frame + 1] = new_eigenvalues

    return dispersion, *lines

animacion = FuncAnimation(fig, func=update, frames=t_animacion * fps, blit=True, interval=1000/fps)
plt.show()
