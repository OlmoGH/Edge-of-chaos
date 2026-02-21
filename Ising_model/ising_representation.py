import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def upload_data():
    data = np.loadtxt("States.txt", dtype=int)
    dim = np.shape(data)[1]
    iterations = np.shape(data)[0] // dim
    states = data.reshape(iterations, dim, dim)

    return states, dim, iterations

def show_ising():
    states, dim, iterations = upload_data()

    fig, ax = plt.subplots()

    cuadro = ax.imshow(np.empty_like(states[0]), cmap='gray', vmin=-1, vmax=1)
    titulo = ax.set_title("Iteración 0")

    def upload(frame):
        titulo.set_text(f"Iteración {frame}")
        cuadro.set_data(states[frame])
        return cuadro, titulo

    animacion = FuncAnimation(fig=fig, func=upload, frames=iterations, blit=True, interval=10)
    plt.show()

def show_time_series():
    states, dim, iterations = upload_data()
    array = np.array([states[i].flatten() for i in range(iterations)]).T
    plt.imshow(array, cmap='gray', vmin=-1, vmax=1, aspect='auto', interpolation='none')
    plt.show()

def show_eigenvalues():
    eigenvalues = np.loadtxt("CovarianceEigenvalues.txt", dtype=float)
    bins = np.logspace(np.log10(eigenvalues.min()), np.log10(eigenvalues.max()), 50)

    counts, edges, _ = plt.hist(
    eigenvalues, 
    bins=bins, 
    log=True, 
    label='Densidad de Autovalores'
    )
    plt.loglog((edges[:-1] + edges[1:]) / 2, counts, 'o', color='red', markersize=5)


    plt.title('Distribución de Autovalores en Escala Log-Log')
    plt.xlabel(r'$\log(\lambda)$ (Autovalor)')
    plt.ylabel(r'$\log(P(\lambda))$ (Densidad/Frecuencia)')
    plt.show()

show_eigenvalues()