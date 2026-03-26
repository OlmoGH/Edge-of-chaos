import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from antiHebbianEvolution import Simulate_and_save as Simulate_hebbian
from antiHebbianMultEvolution import Simulate_and_save as Simulate_Mult_Hebbian
from DataManagement import read_data
from matplotlib.patches import Rectangle

def show(temporal_series, eigenvalues, cov_eigs_sorted, heatmap_min):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))

    # Precalculamos las partes reales e imaginarias para no repetirlo
    real_eigenvalues = np.real(eigenvalues)
    
    # 1️⃣ Temporal series heatmap
    im = ax[0,0].imshow(temporal_series.T, cmap='hot', interpolation='none', aspect='auto', vmin=heatmap_min)
    ax[0,0].set_title("Temporal Series")
    ax[0,0].set_xlabel("Time steps")
    ax[0,0].set_ylabel("Neuron index")

    # 3️⃣ Covariance eigenvalue spectrum
    # Ahora solo plotea, no calcula matemáticas pesadas
    ax[0,1].plot(cov_eigs_sorted, 'o', markersize=3)
    ax[0,1].set_title("Covariance Eigenvalue Spectrum")
    ax[0,1].set_xlabel("Index")
    ax[0,1].set_ylabel("Eigenvalue")
    ax[0,1].grid(True, alpha=0.3)

    # 2️⃣ Eigenvalues in the complex plane (first saved matrix)
    # Reutilizamos el cálculo de la parte real
    ax[1,0].scatter(real_eigenvalues[0], np.imag(eigenvalues[0]), color='blue', s=10)
    ax[1,0].set_title("Eigenvalues (Complex Plane)")
    ax[1,0].set_xlabel("Re(λ)")
    ax[1,0].set_ylabel("Im(λ)")
    ax[1,0].grid(True, alpha=0.3)
    ax[1,0].axhline(0, color='gray', lw=0.5)
    ax[1,0].axvline(0, color='gray', lw=0.5)
    ax[1,0].set_aspect('equal')

    # 4️⃣ Evolution of the real part of the eigenvalues
    # Añadimos rasterized=True. Si tienes más de 100,000 puntos, esto salva la vida de tu CPU.
    ax[1,1].plot(real_eigenvalues, 'b.', markersize=1, rasterized=True)
    ax[1,1].set_title(r"Evolution of the Re($\lambda$)")
    ax[1,1].set_xlabel("Time")
    ax[1,1].set_ylabel("Re(λ)")

    plt.tight_layout(pad=3.0)
    plt.show()

def animar_autovalores(alpha, dim, dt, skip_lote, real, imag):
    skip = 1
    fig, ax = plt.subplots()
    eigvals = ax.scatter([], [], color=color, marker='.', rasterized=True)
    time_txt = ax.text(x=0, y=1.4, s="", ha='center', va='center', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    def update(frame):
        frame = frame//skip
        coords = np.column_stack((real[frame], imag[frame]))
        eigvals.set_offsets(coords)
        time_txt.set_text(f"t = {frame * skip * skip_lote * dt:.0f}")

        return eigvals, time_txt
    animation = FuncAnimation(fig, update, frames=range(0, real.shape[0], skip), blit=False, interval=20)
    plt.show()
    # animation.save(f"Animacion_{alpha}_{dim}")

def mostrar_evolucion_autovalores(real_eigvals_array, imag_eigvals_array, dt, skip):
    print("El tamaño de los arrays es ", real_eigvals_array.shape)

    fig, [ax_re, ax_im] = plt.subplots(ncols=2, figsize=(10, 5))

    pasos = np.arange(real_eigvals_array.shape[0])
    skip = 10
    dt = 0.01
    tiempo = pasos * dt * skip
    ax_re.plot(tiempo, real_eigvals_array, 'b.', markersize=1, rasterized=True)
    ax_im.plot(tiempo, imag_eigvals_array, 'b.', markersize=1, rasterized=True)
    zoom_ax = ax_re.inset_axes([0.5, 0.7, 0.3, 0.3])
    zoom_ax.plot(tiempo, real_eigvals_array, 'b.', markersize=1, rasterized=True)

    ax_re.set_xlabel("Tiempo")
    ax_re.set_ylabel(r"$Re[\lambda(W)]$")
    ax_re.grid(True, linestyle='--', alpha=0.5)
    ax_re.set_xscale('log')

    zoom_ax.set_xlim(50000 * dt * skip, 55000 * dt * skip)
    zoom_ax.set_ylim(-0.3, 0.3)
    zoom_ax.set_xscale('linear')

    ax_re.indicate_inset_zoom(zoom_ax, edgecolor="black", alpha=0.5)
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    ax_im.set_xlabel("Tiempo")
    ax_im.set_ylabel(r"$Im[\lambda(W)]$")
    ax_im.grid(True, linestyle='-', alpha=1)
    ax_im.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()

def actividades_complejas_rasterplot(actividades):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    im = ax[0].imshow(np.transpose(actividades).real, cmap='magma', interpolation='none', aspect='auto')
    ax[0].set_title("Parte real de la actividad neuronal")

    ax[1].imshow(np.transpose(actividades).imag, cmap='magma', interpolation='none', aspect='auto')
    ax[1].set_title("Parte imaginaria de la actividad neuronal")

    plt.colorbar(im, ax=ax)

    plt.show()

def actividades_modulo_rasterplot(actividades):
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))
    im = ax.imshow(np.abs(np.transpose(actividades)), cmap='magma', interpolation='none', aspect='auto')
    ax.set_title("Módulo de la actividad neuronal")

    plt.colorbar(im, ax=ax)

    plt.show()

def actividades_rasterplot(actividades):
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))
    im = ax.imshow(np.transpose(actividades), cmap='magma', interpolation='none', aspect='auto')
    ax.set_title("Parte real de la actividad neuronal")

    plt.colorbar(im, ax=ax)

    plt.show()

def obtener_mejor_frecuencia(lista_señales, steps, skip, dt):
    ventana = np.hanning(steps)

    ncols = 5
    nrows = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 5))
    for i in range(nrows):
        for j in range(ncols):
            señal = lista_señales[i * ncols + j]
            ax[i,j].plot(señal * ventana, rasterized=True)
            fft_mag = np.abs(np.fft.rfft(señal * ventana))
            id_max_freq = np.argmax(fft_mag)
            max_freq_pond = np.sum([i * fft_mag[i] for i in [id_max_freq - 1, id_max_freq, id_max_freq + 1]]) / np.sum([fft_mag[i] for i in [id_max_freq - 1, id_max_freq, id_max_freq + 1]])
            real_freq = max_freq_pond / (steps * skip * dt)
            ax[i, j].set_title(f"f = {real_freq:.5f}", fontsize=10)
            print(f"Frecuencia del autovalor {i * ncols + j}: {real_freq}")

    plt.show()

ALPHA = 1.0E-010
DT = 0.01
DIM = 20
SIMULATED_STEPS = 1_000_000
START = 0
CHUNK_STEPS = 100_000
SKIP = 10
SAVED_STEPS = (SIMULATED_STEPS - START)//SKIP
calc_eigenvalues = False

# Simulamos la red con los parámetros dados
Simulate_Mult_Hebbian(ALPHA, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, START, calc_eigenvalues)

# Leemos los datos de la simulación y calculamos los autovalores (opcional)
archivo_hdf5, X, W, real_eigvals, imag_eigvals = read_data()
print("Datos leidos")

plt.plot(X[:, 0])
plt.show()

# fig, ax = plt.subplots(ncols=2, figsize=(10,  5))
# ax[0].plot(X[:, 0])
# ax[1].plot(real_eigvals, 'b.', markersize=1, rasterized=True)
# plt.show()

# y200_new = np.load("cov_eig_200.npy")
# y500_new = np.load("cov_eig_500.npy")

# plt.loglog(y200_new)
# plt.loglog(y500_new)
# plt.grid(True, alpha=0.8)
# plt.tight_layout()
# plt.show()
# plt.plot(real_eigvals[:], 'b.', markersize=1, rasterized=True)

# cov = np.cov(X[:], rowvar=False)
# eig_cov = np.linalg.eigvalsh(cov)
# np.save(f"cov_eig_{DIM}", eig_cov[::-1])
# plt.plot(eig_cov[::-1])
# plt.show()


# # Calculamos la energía de la red como E = |X|
# energy = np.linalg.norm(X[:], axis=1)**2
# plt.plot(energy)
# plt.show()
# plt.loglog(np.abs(np.fft.rfft(energy - np.mean(energy)))**2)
# plt.show()
# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
# ax[0].plot(real_eigvals[:, 0], 'r', linewidth=2, rasterized=True)
# ax[0].plot(real_eigvals[:, 2], 'b', linewidth=2, rasterized=True)
# ax[1].plot(energy, 'k', rasterized=True)
# ax[0].set_xlim([24750, 26750])
# ax[1].set_xlim([24750, 26750])

archivo_hdf5.close()
