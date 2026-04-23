import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
import scipy.io as sio

def sort_eigenvalues(last_sorted, chunk_eigvals):
    sorted_eigvals = np.zeros_like(chunk_eigvals)
    T, DIM = np.shape(chunk_eigvals)
    if last_sorted is None:
        sorted_eigvals[0] = chunk_eigvals[0]
    else:
        cost_matrix = np.abs(last_sorted[:, None] - chunk_eigvals[0][None, :])
        row_index, col_index = linear_sum_assignment(cost_matrix)
        sorted_eigvals[0] = chunk_eigvals[0][col_index]

    for i in range(1, T):
        cost_matrix = np.abs(sorted_eigvals[i-1][:, None] - chunk_eigvals[i][None, :])
        row_index, col_index = linear_sum_assignment(cost_matrix)
        sorted_eigvals[i] = chunk_eigvals[i][col_index]

    return sorted_eigvals


# ============================================================================
# 1. PARÁMETROS DEL MODELO 
# ============================================================================

np.random.seed(42)

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
Nsteps = TotalSteps // CalcEvery

# ============================================================================
# 2. INICIALIZACIÓN DE VARIABLES Y ESTÍMULOS
# ============================================================================

y = np.zeros(N)                 
z = np.zeros(N)                 
input_vec = np.zeros(N)         
xlp = np.zeros(N)               
B = 0.5 * np.eye(N)             

H = np.sign(np.random.randn(N, N)) 
u = H[0:Nmem, :].T / np.sqrt(N)        
v = H[Nmem:2*Nmem, :].T / np.sqrt(N)   

input1 = np.zeros((N, TotalSteps))
input2 = np.zeros((N, TotalSteps))

for i in range(Nmem):
    base = spacing0 + i * (inLen + spacing)
    for j in range(inLen):
        input1[:, base + j] = u[:, i]
        input2[:, base + j] = v[:, i]

# Cargamos el estado exacto del que parten los autores
print("Cargando condiciones iniciales de data.mat...")
mat_contents = sio.loadmat('data.mat')

try:
    W = mat_contents['data']['W'][0,0]
    x = mat_contents['data']['x'][0,0].flatten()
except KeyError:
    W = mat_contents['W']
    x = mat_contents['x'].flatten()

# Nos aseguramos de que el tipo de dato sea float64 para Numba
W = np.ascontiguousarray(W, dtype=np.float64)
x = np.ascontiguousarray(x, dtype=np.float64)            

# ============================================================================
# 3. BUCLE DE SIMULACIÓN OPTIMIZADO CON NUMBA
# ============================================================================

@njit(fastmath=True)
def EvolveNetwork(x, W, y, xlp, input_vec, input1, input2, B, TotalSteps, CalcEvery, Nsteps, N, dt, taux, tau, zeta, amp, eta):
    
    x_all = np.zeros((N, TotalSteps))
    W_all = np.zeros((N, N, Nsteps))
    idx_W = 0 
    
    # --- NUEVO: Calculamos cada cuántos pasos hacer el print (ej: cada 10%) ---
    print_interval = TotalSteps // 10
    if print_interval == 0:
        print_interval = 1

    for i in range(TotalSteps):
        
        # --- NUEVO: Print de progreso dentro de Numba ---
        if i % print_interval == 0:
            print("Simulando paso", i, "/", TotalSteps, "(", int(i/TotalSteps * 100), "% )")
            
        if i % CalcEvery == 0 and idx_W < Nsteps:
            W_all[:, :, idx_W] = W
            idx_W += 1
            
        x_all[:, i] = x
        r = np.tanh(x)
        xlp = ((-xlp + x / 1e-2) / taux) * dt
        
        noise1 = np.random.randn() 
        noise2 = np.random.randn()
        input_vec = input_vec + (-zeta * input_vec + noise1 * input1[:, i] + noise2 * input2[:, i]) * dt
        
        y = y + (r - y) * dt / tau
        x = x + (-x + np.dot(W, r) + amp * input_vec) * dt
        
        homeostasis = B - np.outer(np.tanh(x - xlp), r)
        mat_noise = np.random.randn(N, N) / np.sqrt(N)
        stdp = np.outer(r, y) - np.outer(y, r)
        
        W = W + eta * (homeostasis + mat_noise + stdp) * dt
        
    return x_all, W_all, W, x

# ============================================================================
# 4. EJECUCIÓN 
# ============================================================================

print("Simulando red... (La primera vez tardará unos segundos en compilar Numba)")
start_time = time.time()

x_all, W_all, W_final, x_final = EvolveNetwork(
    x, W, y, xlp, input_vec, input1, input2, B, 
    TotalSteps, CalcEvery, Nsteps, N, dt, taux, tau, zeta, amp, eta
)

print(f"Simulación completada en {time.time() - start_time:.2f} segundos.")

# ============================================================================
# 5. ANÁLISIS Y GRÁFICA CON SOMBREADO
# ============================================================================

print("Calculando y ordenando autovalores...")
eigenvalues = np.linalg.eigvals(np.transpose(W_all, axes=[2, 0, 1]))
sorted_eigvals = sort_eigenvalues(None, eigenvalues)

# --- NUEVO: Creamos un eje temporal real y calculamos los tiempos de sombreado ---
time_axis = np.arange(1, Nsteps + 1) * CalcEvery * dt
start_learning_time = spacing0 * dt
end_learning_time = (spacing0 + inLen) * dt

plt.figure(figsize=(10, 6))

# Dibujamos las trayectorias de los autovalores imaginarios
plt.plot(time_axis, sorted_eigvals.imag)

# Añadimos la región sombreada donde se inyecta el input (Learning)
plt.axvspan(start_learning_time, end_learning_time, color='red', alpha=0.15, label='Fase de Aprendizaje')

plt.title('Evolución de los autovalores imaginarios')
plt.xlabel('Tiempo (s)')
plt.ylabel('Parte Imaginaria')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()