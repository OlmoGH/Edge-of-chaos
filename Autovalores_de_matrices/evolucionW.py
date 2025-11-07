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
t_animacion = 10000

# Funciones de evolución
def evolucionar_x():
    global x, W
    x = x @ (np.identity(N) + paso * W)

def evolucionar_w():
    global x, W
    W = W + alpha * paso * (np.identity(N) - np.outer(x, x))

# Configuración de gráficos
fig, ax = plt.subplots()
ax.set_aspect('equal')

circulo = Circle((0, 0), radius=np.sqrt(N), fill=False)
ax.add_patch(circulo)
dispersion = ax.scatter([], [], s=10)

# Función de actualización
def update(frame):
    frames = np.linspace(0, frame * paso, frame + 1)

    eigenvalues = np.linalg.eigvals(W)

    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)

    dispersion.set_offsets(np.c_[real, imag])

    evolucionar_x()
    evolucionar_w()

    return dispersion, 

animacion = FuncAnimation(fig, func=update, frames=t_animacion * fps, blit=True, interval=1000/fps)
animacion.save("Animacion.mp4", fps=70)
