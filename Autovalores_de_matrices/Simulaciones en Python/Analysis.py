import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import AntiHebbianMejorado
from DataManagement import read_data
from scipy.linalg import null_space
import STDP
import STDP_susman
import STDP_Euler
from numba import njit

@njit(fastmath=True)
def get_tracked_eigenvectors(W_seq, target_eigenvalues):
    T = W_seq.shape[0]
    DIM = W_seq.shape[1]
    
    # Extraemos cuántos autovalores estamos siguiendo simultáneamente
    N_eigenvalues = target_eigenvalues.shape[1]
    
    # Nueva forma: (Tiempo, Número de targets, Dimensión del vector)
    eigenvectors = np.zeros((T, N_eigenvalues, DIM), dtype=np.complex128)
    
    for i in range(T):
        if i % (T // 10) == 0: 
            print("Paso ", i, " / ", T)
            
        # 1. Calculamos TODOS los autovalores y autovectores de golpe
        # FORZAMOS el paso a complex128 para evitar el bug de desempaquetado de Numba
        vals, vecs = np.linalg.eig(W_seq[i].astype(np.complex128))
        
        # 2. Para CADA autovalor que queremos rastrear, buscamos su pareja
        for k in range(N_eigenvalues):
            target_val = target_eigenvalues[i, k]
            
            min_dist = 1e9
            best_idx = 0
            
            for j in range(DIM):
                # abs() con complejos calcula el módulo (distancia euclídea)
                dist = np.abs(vals[j] - target_val)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = j
                    
            # 3. Guardamos el autovector correspondiente
            # vecs[:, best_idx] es la columna con el vector
            # Lo guardamos en la fila correspondiente a nuestro target 'k'
            eigenvectors[i, k, :] = vecs[:, best_idx]
            
    print("Acabé")
    return eigenvectors

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

def animar_autovalores(skip_frames, dt, skip_lote, real, imag, save=False):
    fig, ax = plt.subplots()
    eigvals, = ax.plot([], [], 'b.', rasterized=True)
    time_txt = ax.text(x=0, y=1.4, s="", ha='center', va='center', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim([1.2 * real[:].min(), 1.2 * real[:].max()])
    ax.set_ylim([1.2 * imag[:].min(), 1.2 * imag[:].max()])

    def update(frame):
        eigvals.set_data(real[frame], imag[frame])
        time_txt.set_text(f"t = {frame * skip_lote * dt:.0f}")

        return eigvals, time_txt
    animation = FuncAnimation(fig, update, frames=range(0, real.shape[0], skip_frames), blit=False, interval=20)
    if save: animation.save(f"Animacion_{ALPHA}_{DIM}.gif")
    plt.show()

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

#-------------------------------------------------------------------------

#################################
## PARÁMETROS DE LA SIMULACIÓN ##
#################################

ALPHA = 0.001
TAU = 50
DT = 0.1
DIM = 128
# Pasos simulados y guardados
SIMULATED_STEPS = 200_000
# Pasos previos para el warmup
CHUNK_STEPS = 100_000
SKIP = 10
SAVED_STEPS = SIMULATED_STEPS//SKIP
calc_eigenvalues = True

#Inicializamos la matriz de conexiones y el vector de neuronas
W = np.random.normal(0, 1.0/np.sqrt(DIM), (DIM, DIM))
X = np.random.normal(0, 1.0, DIM)

INPUT_X_RANGE = [150_000, 160_000]
X_in = INPUT_X_RANGE[0]
X_out = INPUT_X_RANGE[1]
rng_u = np.random.default_rng(42)
rng_v = np.random.default_rng(69)

u = np.sign(rng_u.standard_normal(DIM)) / np.sqrt(DIM)
v = np.sign(rng_v.standard_normal(DIM)) / np.sqrt(DIM)
input_u = np.zeros((SIMULATED_STEPS, DIM))
input_v = np.zeros((SIMULATED_STEPS, DIM))
input_u[X_in:X_out] = u
input_v[X_in:X_out] = v

# Tomamos los coeficientes de un prceso de Ornstein-Uhlenbeck
tau_ou = 20.0  # tiempo de correlación en unidades de tiempo
dt = DT

INPUT_X = np.zeros((SIMULATED_STEPS, DIM))
for i in range(1, SIMULATED_STEPS):
    INPUT_X[i] = INPUT_X[i-1] + (-INPUT_X[i-1]/tau_ou + np.random.randn()*input_u[i] + np.random.randn()*input_v[i]) * DT * 25
plt.plot(INPUT_X)
plt.show()
#-------------------------------------------------------------------------

#######################
# SIMULACIÓN #
#######################

# Simulamos la red con los parámetros dados
STDP_Euler.Simulate_and_save_STDP(X, W, INPUT_X, ALPHA, TAU, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, calc_eigenvalues)

#-------------------------------------------------------------------------

####################
# LECTURA DE DATOS #
####################

# Leemos los datos de la simulación y calculamos los autovalores (opcional)
archivo_hdf5, X, W, real_eigvals, imag_eigvals = read_data()
print("Datos leidos")

#-------------------------------------------------------------------------

#############################
# ANÁLISIS DE LA SIMULACIÓN #
#############################

energy = np.linalg.norm(X[:], axis=1)
plt.plot(energy)
plt.show()
p_u = np.dot(X[:], u)
p_v = np.dot(X[:], v)

p_uv = np.sqrt(p_u**2 + p_v**2)

w = np.random.standard_normal(DIM)
w = w / np.linalg.norm(w)
p_w = np.dot(X[:], w)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(imag_eigvals)
ax[1].plot(real_eigvals)
plt.show()
plt.plot(p_uv, alpha=0.8, label=r"$P_{uv}$")
plt.plot(np.abs(p_w), alpha=0.8, label=r"$P_w$")
plt.legend()
plt.show()

archivo_hdf5.close()