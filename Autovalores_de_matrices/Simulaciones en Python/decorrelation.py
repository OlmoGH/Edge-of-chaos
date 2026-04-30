import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import scipy.io as sio  # <-- NUEVA IMPORTACIÓN PARA LEER .mat
import sys
import os

def sort_eigenvalues(last_sorted, current_eigvals):
    """
    Función auxiliar para hacer el seguimiento (eigenshuffle) de los autovalores 
    a través del tiempo asociando los más cercanos.
    """
    if last_sorted is None:
        return current_eigvals
    cost_matrix = np.abs(last_sorted[:, None] - current_eigvals[None, :])
    row_index, col_index = linear_sum_assignment(cost_matrix)
    return current_eigvals[col_index]

# %%%% Carga de datos pre-inicializados desde data.mat
# Buscamos 'data.mat' en el mismo directorio que este script
directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
ruta_mat = os.path.join(directorio_actual, 'data.mat')

# Cargamos el archivo. 
# squeeze_me y struct_as_record nos permiten usar la sintaxis "data.W" en lugar de diccionarios anidados complejos.
mat_contents = sio.loadmat(ruta_mat, squeeze_me=True, struct_as_record=False)

# Extraemos la estructura 'data'
data = mat_contents['data']

# Fijamos la semilla exactamente como en MATLAB
# Comprobamos si es una estructura de MATLAB (estado del RNG) o un número directo
try:
    seed = int(data.decseed.Seed)
except AttributeError:
    seed = int(data.decseed)

np.random.seed(seed)

# %%%% Parámetros iniciales
N = 128
Nmem = 1
dt = 0.1
eta = 0.01
tau = 50.0
zeta = 0.01
amp = 25.0
inLen = int(100 / dt)
spacing = int(1000 / dt)

# --- MODIFICACIÓN AQUÍ ---
# Aumentamos spacing0 para que el input ocurra más tarde. 
# Antes era 200, ahora es 800 (empezará en el tiempo t=800).
spacing0 = int(800 / dt) 
# -------------------------

TotalSteps = spacing0 + (inLen + spacing) * Nmem
CalcEvery = int(10 / dt)
Nsteps = TotalSteps // CalcEvery

# Inicialización de variables
y = np.zeros(N)
z = np.zeros(N)
x_all = np.zeros((N, TotalSteps))

H = np.sign(np.random.randn(N, N))
u = H[0:Nmem, :].T / np.sqrt(N)
v = H[Nmem:2*Nmem, :].T / np.sqrt(N)

input1 = np.zeros((N, TotalSteps))
input2 = np.zeros((N, TotalSteps))
W_all = np.zeros((Nsteps, N, N))
inp = np.zeros(N)
B = np.eye(N)

# --- CARGAMOS W Y X DESDE EL ARCHIVO MAT ---
W = data.W.copy()
x = data.x.copy()
# -------------------------------------------

print('Simulating, please wait...')

# %%%% Construct input signal for learning
for i in range(Nmem):
    base = i * spacing + spacing0
    input1[:, base:base+inLen] = np.tile(u[:, i:i+1], (1, inLen))
    input2[:, base:base+inLen] = np.tile(v[:, i:i+1], (1, inLen))

# %%%% Evolve network
for i in range(TotalSteps):
    
    if i % CalcEvery == 0:
        W_all[i // CalcEvery] = W.copy()
        if (i // CalcEvery) % (Nsteps // 10) == 0:
            sys.stdout.write(f"\rProgreso: {(i/TotalSteps)*100:.0f}%")
            sys.stdout.flush()
            
    x_all[:, i] = x.copy()
    
    # Ruido de entrada con Ornstein-Uhlenbeck 
    inp = inp + (-inp * zeta + np.random.randn() * input1[:, i] + np.random.randn() * input2[:, i]) * dt
    
    dxdt = np.dot(W, x) + amp * inp
    dydt = (x - y) / tau  
    
    # Actualización de Euler en cascada / secuencia
    noise_W = np.random.randn(N, N) / np.sqrt(N)
    dW = (B - np.outer(x, x) + noise_W) + (np.outer(x, y) - np.outer(y, x))
    W = W + eta * dW * dt
    
    x = x + dxdt * dt
    y = y + dydt * dt

print("\rProgreso: 100%")

# %%%% Compute eigenspectrum of W over time
Dseq = np.zeros((Nsteps, N), dtype=complex)
last_sorted = None
for i in range(Nsteps):
    eigvals = np.linalg.eigvals(W_all[i])
    sorted_eigvals = sort_eigenvalues(last_sorted, eigvals)
    Dseq[i] = sorted_eigvals
    last_sorted = sorted_eigvals

# Transponemos Dseq para que sea (N x Nsteps)
Dseq = Dseq.T

I = np.argsort(np.imag(Dseq[:, -1]))[::-1]
taxis = np.arange(1, Nsteps + 1) * CalcEvery * dt

# %%%% Plot spectrum
plt.figure(figsize=(10, 8))

# Subplot 1: Real Part
plt.subplot(2, 1, 1)
for j in range(N):
    plt.plot(taxis, np.real(Dseq[I[j], :]), linewidth=2, color=[0.8, 0.8, 0.8])
plt.plot(taxis, np.real(Dseq[I[0], :]), linewidth=2, color=[0.47, 0.67, 0.19])
plt.plot(taxis, np.real(Dseq[I[-1], :]), linewidth=2, color=[0.47, 0.67, 0.19])
plt.ylabel('Re($\lambda$)', fontsize=14)
plt.tick_params(labelsize=12)

# Subplot 2: Imaginary Part
plt.subplot(2, 1, 2)
for j in range(N):
    plt.plot(taxis, np.imag(Dseq[I[j], :]), linewidth=2, color=[0.8, 0.8, 0.8])
plt.plot(taxis, np.imag(Dseq[I[0], :]), linewidth=2, color=[0.47, 0.67, 0.19])
plt.plot(taxis, np.imag(Dseq[I[-1], :]), linewidth=2, color=[0.47, 0.67, 0.19])
plt.xlabel('time', fontsize=14)
plt.ylabel('Im($\lambda$)', fontsize=14)
plt.tick_params(labelsize=12)

plt.tight_layout()
plt.show()