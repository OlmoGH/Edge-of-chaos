import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

def get_eigenvalues(alpha, dim, dt):
    # Lectura de los datos y conversión autovalores = [[re1, im1], [re2, im2], ...]
    raw_data = np.fromfile(f'output/Eigenvalues_{alpha}_{dim}_{dt}.bin', dtype=np.complex128)
    data = raw_data.reshape((-1, dim))
    print(data[0])

    # Ordenamos los autovalores por su parte real en cada iteración para que las variaciones en cada autovalor sean suaves
    sorted_indices = np.argsort(data[0])
    sorted_data = data[:, sorted_indices]
    return sorted_data

def show_eigenvalue_evolution(alpha, dim, dt, show_animation=True):

    eigvals = get_eigenvalues(alpha, dim, dt)
    # Gráfica inicial de los autovalores
    plt.plot(np.real(eigvals), '.', markersize=2)
    plt.gca().set_xscale('linear')
    plt.show()

    if show_animation:
        # Animación de los autovalores

        fig, [ax_line, ax_scat] = plt.subplots(ncols=2, figsize=[10, 5])
        ax_scat.set_xlim(-1.2, 1.2)
        ax_scat.set_ylim(-1.2, 1.2)
        ax_scat.grid(True, alpha=0.3)
        ax_scat.set_aspect('equal')


        ax_line.set_xscale('log')
        ax_line.set_xlim([10e0, len(eigvals)])
        ax_line.set_ylim([-1.2, 1.2])
        ax_line.set_ylabel("Re[eig(W)]")
        ax_line.set_xlabel("t")

        lines = [ax_line.plot([], [], 'b.', markersize=0.5)[0] for i in range(dim)]

        points = ax_scat.scatter(x=[], y=[], c='blue')
        circle = Circle(xy=[0, 0], radius=1, fill=False)
        ax_scat.add_patch(circle)

        fig.suptitle(fr"Evolución de los autovalores: dim = {dim}, $\alpha$ = {alpha}, iteraciones = {len(eigvals)}")
        ax_scat.set_title("Dinámica de los autovalores en el plano complejo")
        ax_line.set_title("Evolución de la parte real de los autovalores")

        def update(frame):
            curr_frame = frame * skip
            points.set_offsets(np.transpose([np.real(eigvals[curr_frame]), np.imag(eigvals[curr_frame])]))
            [lines[i].set_data(np.arange(curr_frame), np.real(eigvals[:curr_frame, i])) for i in range(dim)]

            return points, *lines

        animation = FuncAnimation(fig=fig, func=update, frames=len(eigvals)//(skip), blit=True, interval=10)
        plt.show()

# Parámetros de la simulación necesarios para localizar el archivo de datos
alpha = 0.01
dim = 30
dt = 0.01
skip = 100

show_eigenvalue_evolution(alpha, dim, dt, True)