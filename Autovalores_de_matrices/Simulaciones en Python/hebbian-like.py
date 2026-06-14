import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from eigenshuffle import eigenshuffle_eig
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import signal
from scipy.linalg import expm
from tqdm import tqdm

# 1. Creamos una pequeña función auxiliar para determinantes 2x2 a mano
@njit(fastmath=True)
def _det2x2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

# 2. La función principal optimizada para Numba
@njit(fastmath=True)
def _calcular_overlaps_numba(u, v, all_evals, all_evecs):
    dim = u.shape[0]
    T = all_evals.shape[0]
    overlaps = np.zeros(T)

    # Creamos la matriz U de forma Numba-friendly (sin column_stack)
    U = np.zeros((dim, 2), dtype=np.complex128)
    U[:, 0] = u
    U[:, 1] = v
    
    # Determinante de UU (es real y positivo, aplicamos abs por precisión numérica)
    UU = np.abs(_det2x2(U.T.conj() @ U))

    # Pre-asignamos la matriz R una sola vez fuera del bucle para ahorrar memoria
    R = np.zeros((dim, 2), dtype=np.complex128)

    for t in range(T):
        evals = all_evals[t]
        evecs = all_evecs[t]

        # Ordenamos los índices por la parte imaginaria
        idx = np.argsort(evals.imag)
        
        # Asignamos las columnas directamente a la matriz R existente
        R[:, 0] = evecs[:, idx[0]]   # Menor parte imaginaria
        R[:, 1] = evecs[:, idx[-1]]  # Mayor parte imaginaria

        # Multiplicaciones de matrices
        gram_RR = R.T.conj() @ R
        gram_UR = U.T.conj() @ R

        # Calculamos determinantes de 2x2 a la velocidad del rayo
        RR = np.abs(_det2x2(gram_RR))
        UR = _det2x2(gram_UR)

        # Guardamos el overlap absoluto
        overlaps[t] = np.abs(UR) / np.sqrt(RR * UU)

    return overlaps

# 3. Función envoltorio (Wrapper) que usarás en tu código principal
def overlap_uv_max_imag(u, v, W):
    T = W.shape[0]
    dim = W.shape[1]
    
    # Pre-calculamos los autovalores y autovectores en NumPy (soporta complex128)
    all_evals = np.zeros((T, dim), dtype=np.complex128)
    all_evecs = np.zeros((T, dim, dim), dtype=np.complex128)
    
    # Aseguramos que W sea complejo
    W_complex = W.astype(np.complex128)
    
    for t in tqdm(range(T)):
        all_evals[t], all_evecs[t] = np.linalg.eig(W_complex[t])
        
    # Le pasamos los arrays cocinados a Numba
    return _calcular_overlaps_numba(u, v, all_evals, all_evecs)

def track_eigsystem_fixed(matrices):
    T, N, _ = matrices.shape
    tracked_eigvals = np.zeros((T, N), dtype=np.complex128)
    # Guardaremos los autovectores de forma que tracked_eigvecs[t, :, i] sea el i-ésimo vector
    tracked_eigvecs = np.zeros((T, N, N), dtype=np.complex128)
    
    # --- Paso t=0 ---
    eigvals_initial, eigvecs_initial = np.linalg.eig(matrices[0])
    sorted_indices = np.argsort(eigvals_initial.real)
    
    tracked_eigvals[0] = eigvals_initial[sorted_indices]
    tracked_eigvecs[0] = eigvecs_initial[:, sorted_indices] # <-- CORREGIDO: Indexación por columnas
    
    if T == 1:
        return tracked_eigvals, tracked_eigvecs

    # --- Pasos t > 0 ---
    for t in tqdm(range(1, T)):
        curr_eigs, curr_eigvecs = np.linalg.eig(matrices[t])
        
        # Calculamos el producto escalar entre los vectores anteriores y los nuevos.
        # dot_products[i, j] = producto escalar del vector i (viejo) con el vector j (nuevo)
        dot_products = np.dot(tracked_eigvecs[t-1].T.conj(), curr_eigvecs)
        
        # Definimos la matriz de costo: queremos maximizar el valor absoluto del solape.
        # Costo mínimo = Máximo solape (cercano a 1)
        cost_matrix = 1.0 - np.abs(dot_products)
        
        # Resolver el problema de asignación lineal
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        # Guardamos autovalores y autovectores reordenados correctamente
        tracked_eigvals[t] = curr_eigs[col_ind]
        tracked_eigvecs[t] = curr_eigvecs[:, col_ind] # <-- CORREGIDO: Columnas
        
        # --- CORRECCIÓN DE FASE/SIGNO ---
        # Forzamos a que el autovector actual apunte en la misma dirección/fase que el anterior
        for i in range(N):
            # Calculamos el desajuste de fase exacto
            v_old = tracked_eigvecs[t-1, :, i]
            v_new = tracked_eigvecs[t, :, i]
            dot = np.vdot(v_old, v_new)
            
            if np.abs(dot) > 1e-9:
                phase = dot / np.abs(dot)
                # Multiplicamos por el conjugado de la fase para alinearlos
                tracked_eigvecs[t, :, i] *= np.conj(phase)
                
    return tracked_eigvals, tracked_eigvecs

@njit(fastmath=True)
def linear_sum_assignment_numba(cost_matrix):
    n, m = cost_matrix.shape
    
    # Creamos una matriz de costos indexada en 1 (1-indexed) para facilitar el algoritmo
    C = np.zeros((n + 1, m + 1), dtype=np.float64)
    C[1:, 1:] = cost_matrix
    
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=np.bool_)
        
        while p[j0] != 0:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                        
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
                    
            j0 = j1
            
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            
    # Extraemos los resultados al formato que devuelve SciPy
    row_ind = np.arange(n)
    col_ind = np.zeros(n, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            col_ind[p[j] - 1] = j - 1
            
    return row_ind, col_ind

@njit(fastmath=True)
def track_eigenvalues_numba(matrices):
    T, N, _ = matrices.shape
    tracked_eigvals = np.zeros((T, N), dtype=np.complex128)
    matrices = matrices.astype(np.complex128)
    
    # --- Paso t=0 ---
    eigvals_initial = np.linalg.eigvals(matrices[0])
    
    # En Numba, argsort funciona sin problema sobre la parte real
    idx_sort = np.argsort(eigvals_initial.real)
    tracked_eigvals[0] = eigvals_initial[idx_sort]
    
    if T == 1:
        return tracked_eigvals

    # --- Paso t=1 (Aún no tenemos velocidad para predecir) ---
    curr_eigs = np.linalg.eigvals(matrices[1])
    
    # Uso de reshape para asegurar el broadcasting correcto en Numba
    cost_matrix = np.abs(tracked_eigvals[0].reshape(-1, 1) - curr_eigs.reshape(1, -1))
    
    _, col_ind = linear_sum_assignment_numba(cost_matrix)
    tracked_eigvals[1] = curr_eigs[col_ind]

    # --- Pasos t > 1 (Uso de extrapolación predictiva) ---
    for t in range(2, T):
        if t % int(0.1 * T) == 0:
            print("Paso", t, "/", T)

        curr_eigs = np.linalg.eigvals(matrices[t])
        
        # 1. Calculamos la "velocidad" (cuánto se movieron en el último paso)
        velocidad = tracked_eigvals[t-1] - tracked_eigvals[t-2]
        
        # 2. Predecimos dónde deberían estar ahora
        prediccion = tracked_eigvals[t-1] + velocidad
        
        # 3. Emparejamos los actuales con la PREDICCIÓN
        cost_matrix = np.abs(prediccion.reshape(-1, 1) - curr_eigs.reshape(1, -1))
        _, col_ind = linear_sum_assignment_numba(cost_matrix)
        
        # Guardamos el resultado ordenado
        tracked_eigvals[t] = curr_eigs[col_ind]
        
    return tracked_eigvals

def track_eigenvalues(matrices):
    T, N, _ = matrices.shape
    tracked_eigvals = np.zeros((T, N), dtype=np.complex128)
    matrices = matrices.astype(np.complex128)
    
    # --- Paso t=0 ---
    eigvals_initial = np.linalg.eigvals(matrices[0])
    
    # En Numba, argsort funciona sin problema sobre la parte real
    idx_sort = np.argsort(eigvals_initial.real)
    tracked_eigvals[0] = eigvals_initial[idx_sort]
    
    if T == 1:
        return tracked_eigvals

    # --- Paso t=1 (Aún no tenemos velocidad para predecir) ---
    curr_eigs = np.linalg.eigvals(matrices[1])
    
    # Uso de reshape para asegurar el broadcasting correcto en Numba
    cost_matrix = np.abs(tracked_eigvals[0].reshape(-1, 1) - curr_eigs.reshape(1, -1))
    
    _, col_ind = linear_sum_assignment_numba(cost_matrix)
    tracked_eigvals[1] = curr_eigs[col_ind]

    # --- Pasos t > 1 (Uso de extrapolación predictiva) ---
    for t in tqdm(range(2, T)):
        if t % int(0.1 * T) == 0:
            print("Paso", t, "/", T)

        curr_eigs = np.linalg.eigvals(matrices[t])
        
        # 1. Calculamos la "velocidad" (cuánto se movieron en el último paso)
        velocidad = tracked_eigvals[t-1] - tracked_eigvals[t-2]
        
        # 2. Predecimos dónde deberían estar ahora
        prediccion = tracked_eigvals[t-1] + velocidad
        
        # 3. Emparejamos los actuales con la PREDICCIÓN
        cost_matrix = np.abs(prediccion.reshape(-1, 1) - curr_eigs.reshape(1, -1))
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        # Guardamos el resultado ordenado
        tracked_eigvals[t] = curr_eigs[col_ind]
        
    return tracked_eigvals

@njit(fastmath=True)
def hopfield(STEPS, W, X_0, DT):
    DIM = X_0.shape[0]
    X_time_series = np.zeros((STEPS, DIM))
    X_time_series[0] = X_0
    for t in range(1, STEPS):
        if t % 10 == 0:
            print(f"Paso {t}/{STEPS}")
        for i in range(DIM):
            sum = 0
            for j in range(DIM):
                sum += W[i, j] * X_time_series[t-1, j]

            X_time_series[t, i] = np.tanh(sum)
    
    return X_time_series

@njit(fastmath=True)
def lineal_network(STEPS, Evol_matrix, X_0):
    DIM = X_0.shape[0]
    X_time_series = np.zeros((STEPS, DIM))
    X_time_series[0] = X_0
    
    for t in range(1, STEPS):
        if t % int(0.1 * STEPS) == 0:
            print("Paso", t, "/", STEPS)
            
        # Multiplicación matriz-vector exacta (equivalente a Evol_matrix @ X)
        # np.dot está totalmente soportado por Numba y es ultrasónico
        X_time_series[t] = np.dot(Evol_matrix, X_time_series[t-1])
    
    return X_time_series

@njit(fastmath=True)
def anti_hebbian_network_rk4(STEPS, W_0, X_0, DT, ALPHA):
    DIM = X_0.shape[0]
    X_time_series = np.zeros((STEPS, DIM))
    W_time_series = np.zeros((STEPS, DIM, DIM))
    X_time_series[0] = X_0
    W_time_series[0] = W_0

    I = np.eye(DIM)
    
    for t in range(1, STEPS):
        if t % int(0.1 * STEPS) == 0:
            print("Paso", t, "/", STEPS)
            
        x = X_time_series[t-1]
        w = W_time_series[t-1]
        
        k1_X = np.dot(w, x)
        k1_W = ALPHA * (I - np.outer(x, x))
        
        x2 = x + 0.5 * DT * k1_X
        W2 = w + 0.5 * DT * k1_W
        k2_X = np.dot(W2, x2)
        k2_W = ALPHA * (I - np.outer(x2, x2))
        
        x3 = x + 0.5 * DT * k2_X
        W3 = w + 0.5 * DT * k2_W
        k3_X = np.dot(W3, x3)
        k3_W = ALPHA * (I - np.outer(x3, x3))
        
        x4 = x + DT * k3_X
        W4 = w + DT * k3_W
        k4_X = np.dot(W4, x4)
        k4_W = ALPHA * (I - np.outer(x4, x4))
        
        X_time_series[t] = x + (DT / 6.0) * (k1_X + 2.0 * k2_X + 2.0 * k3_X + k4_X)
        W_time_series[t] = w + (DT / 6.0) * (k1_W + 2.0 * k2_W + 2.0 * k3_W + k4_W)
        
    return X_time_series, W_time_series

@njit(fastmath=True)
def anti_hebbian_euler_learning(STEPS, W_0, X_0, DT, ALPHA, input_W):
    dim = X_0.shape[0]
    X = np.zeros((STEPS, dim))
    W = np.zeros((STEPS, dim, dim))
    dXdt = np.zeros(dim)
    dWdt = np.zeros((dim, dim))

    X[0] = X_0
    W[0] = W_0

    I = np.eye(dim)
    for t in range(1, STEPS):
        if t % int(0.1 * STEPS) == 0:
            print("Paso", t, "/", STEPS)
        for i in range(dim):
            dXdt[i] = 0
            for j in range(dim):
                dXdt[i] += W[t-1, i, j] * X[t-1, j]
                dWdt[i, j] = ALPHA * (I[i, j] - X[t-1, i] * X[t-1, j] + input_W[t-1, i, j])
        
        for i in range(dim):
            X[t, i] = X[t-1, i] + DT * dXdt[i]
            for j in range(dim):
                W[t, i, j] = W[t-1, i, j] + DT * dWdt[i, j]
    
    return X, W
        
@njit(fastmath=True)
def evolucion_modificada(steps, X_0, dt, alpha):
    dim = X_0.shape[0]
    X_time_series = np.zeros((steps, dim))
    X_time_series[0] = X_0
    
    for t in range(1, steps):
        if t % int(0.1 * steps) == 0:
            print("Paso", t, "/", steps)
            
        X_time_series[t] = X_time_series[t-1] + dt * (alpha * X_time_series[t-1] - np.linalg.norm(X_time_series[t-1])**2 * X_time_series[t-1])
    return X_time_series


dim = 20
time = 20000
dt = 0.1
steps = int(time/dt)
t_eval = np.linspace(0, time, steps)
alpha = 0.001
threshold = 4
rho = 5

np.random.seed(45432)
W_0 = np.random.randn(dim, dim)
X_0 = np.random.randn(dim)

u = np.random.standard_normal(dim)
u = u / np.linalg.norm(u)

v = np.random.standard_normal(dim)
v = v - np.dot(u, v) * u
v = v / np.linalg.norm(v)

A = rho * (np.outer(u, v) - np.outer(v, u))

input_W = np.zeros((steps, dim, dim))
inicio = int(2 * np.sqrt(dim)/(alpha * dt))
duracion = int(1000 / dt)

input_W[inicio:inicio+duracion] = A

print("Simulando ...")
X, W = anti_hebbian_euler_learning(steps, W_0, X_0, dt, alpha, input_W)
print("Simulación terminada")

# print("Calculando autovalores ...")
# eigenvalues, eigenvectors = track_eigsystem_fixed(W)
# print("Autovalores calculados")

# Comprobamos el overlapping entre el plano uv y el de los autovectores con parte imaginaria mayor
print("Calculando overlaps ...")
overlaps = overlap_uv_max_imag(u, v, W)
print("Overlaps calculados")

plt.plot(overlaps)
plt.axvspan(inicio, inicio+duracion, alpha=0.5)
plt.show()

eigenvalues = track_eigenvalues_numba(W)

fig, ax = plt.subplots(ncols=2)

ax[0].plot(eigenvalues.real)
ax[1].plot(eigenvalues.imag)

ax[0].axvspan(inicio, inicio+duracion, alpha=0.5)
ax[1].axvspan(inicio, inicio+duracion, alpha=0.5)

plt.show()
