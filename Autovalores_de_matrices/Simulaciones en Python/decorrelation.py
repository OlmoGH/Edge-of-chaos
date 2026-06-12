import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange

# --- Parámetros e Inicialización ---
try:
    # 'squeeze_me=True' ayuda a limpiar dimensiones vacías de MATLAB
    mat_contents = loadmat('Autovalores_de_matrices/Simulaciones en Python/data.mat', squeeze_me=True)
    
    # Accedemos al struct llamado 'data'
    mat_data_struct = mat_contents['data']
    
    # --- DESEMPAQUETADO SEGURO ---
    W = mat_data_struct['W']
    if W.dtype == object: 
        W = W.item()  # Saca la matriz real si viene envuelta como objeto
    W = np.asarray(W, dtype=float)  # Forzamos a que sea un array numérico puro

    x = mat_data_struct['x']
    if x.dtype == object: 
        x = x.item()  # Saca el vector real si viene envuelto
    x = np.asarray(x, dtype=float).flatten()  # Forzamos a vector numérico 1D
    
    print(f"¡Archivo data.mat cargado con éxito! Formato de W: {W.shape}")

except KeyError:
    mat_contents = loadmat('Autovalores_de_matrices/Simulaciones en Python/data.mat')
    variables_reales = [k for k in mat_contents.keys() if not k.startswith('__')]
    print(f"Error: No se encontró el struct 'data'. Variables en el archivo: {variables_reales}")
    raise

except FileNotFoundError:
    print("Aviso: No se encontró 'data.mat'. Inicializando de forma aleatoria.")
    N_test = 128
    W = np.random.randn(N_test, N_test) / np.sqrt(N_test)
    x = np.random.randn(N_test)

N = 128
Nmem = 1
dt = 0.1
eta = 0.01
tau = 50.0
taux = 20.0
zeta = 0.01
amp = 25.0

inLen = int(100 / dt)
spacing = int(1000 / dt)
spacing0 = int(200 / dt)
TotalSteps = spacing0 + (inLen + spacing) * Nmem
CalcEvery = int(10 / dt)
Nsteps = int(TotalSteps / CalcEvery)

y = np.zeros(N)
x_all = np.zeros((N, TotalSteps))

# En MATLAB sign(randn(N)) crea una matriz NxN. Extraemos u y v.
H = np.sign(np.random.randn(N, N))
u = H[0:Nmem, :].T / np.sqrt(N)
v = H[Nmem:2*Nmem, :].T / np.sqrt(N)

# Aplanamos a 1D para que encaje mejor con numpy
u = u.flatten()
v = v.flatten()

input1 = np.zeros((N, TotalSteps))
input2 = np.zeros((N, TotalSteps))
W_all = np.zeros((N, N, Nsteps))
input_val = np.zeros(N)
xlp = np.zeros(N)  # En MATLAB era 0, aquí lo inicializamos como vector de 0s
B = 0.5 * np.eye(N)

# --- Construcción de la señal de entrada (Learning) ---
for i in range(Nmem):
    base = i * spacing + spacing0
    # tile repite el vector a lo largo de las columnas (inLen veces)
    input1[:, base:base+inLen] = np.tile(u[:, None], (1, inLen))
    input2[:, base:base+inLen] = np.tile(v[:, None], (1, inLen))

# --- Evolución de la Red ---
# Usamos trange de tqdm para mostrar la barra de progreso
for i in trange(TotalSteps, desc='Simulando red...'):
    
    if i % CalcEvery == 0:
        idx = i // CalcEvery
        if idx < Nsteps:
            W_all[:, :, idx] = W
            
    x_all[:, i] = x
    r = np.tanh(x)
    
    # Ecuaciones diferenciales (Euler)
    xlp = xlp + ((-xlp + x / 1e-2) / taux) * dt
    input_val = input_val + (-zeta * input_val + np.random.randn() * input1[:, i] + np.random.randn() * input2[:, i]) * dt
    
    y = y + (r - y) * dt / tau  
    x = x + (-x + W @ r + amp * input_val) * dt
    
    # Actualización homeostática de la matriz de pesos
    noise_W = np.random.randn(N, N) / np.sqrt(N)
    term1 = B - np.outer(np.tanh(x - xlp), np.tanh(x)) + noise_W
    term2 = np.outer(r, y) - np.outer(y, r)
    W = W + eta * (term1 + term2) * dt

# --- Cálculo del espectro dinámico de W (Eigenshuffle alternativo) ---
print("Calculando y ordenando autovalores...")
Dseq = np.zeros((N, Nsteps), dtype=complex)

for k in range(Nsteps):
    eigvals = np.linalg.eigvals(W_all[:, :, k])
    # Ordenamos de mayor a menor por su parte imaginaria para dar continuidad visual
    sorted_indices = np.argsort(eigvals.imag)[::-1]
    Dseq[:, k] = eigvals[sorted_indices]

taxis = np.arange(1, Nsteps + 1) * CalcEvery * dt

# --- Representación Gráfica (Plot) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Pintamos todas las trazas de fondo en gris
for j in range(N):
    ax1.plot(taxis, Dseq[j, :].real, linewidth=2, color=[0.8, 0.8, 0.8])
    ax2.plot(taxis, Dseq[j, :].imag, linewidth=2, color=[0.8, 0.8, 0.8])

# Resaltamos en verde los extremos (las de mayor parte imaginaria)
ax1.plot(taxis, Dseq[0, :].real, linewidth=2, color=[0.47, 0.67, 0.19])
ax1.plot(taxis, Dseq[-1, :].real, linewidth=2, color=[0.47, 0.67, 0.19])

ax2.plot(taxis, Dseq[0, :].imag, linewidth=2, color=[0.47, 0.67, 0.19])
ax2.plot(taxis, Dseq[-1, :].imag, linewidth=2, color=[0.47, 0.67, 0.19])

# Formateo visual
ax1.set_ylabel(r'Re($\lambda$)', fontsize=14)
ax1.tick_params(labelsize=12)
ax1.set_xlim(taxis[0], taxis[-1])

ax2.set_xlabel('time', fontsize=14)
ax2.set_ylabel(r'Im($\lambda$)', fontsize=14)
ax2.tick_params(labelsize=12)
ax2.set_xlim(taxis[0], taxis[-1])

plt.tight_layout()
plt.show()