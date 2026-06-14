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
    for t in range(1, T):
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

def track_eigsystem(matrices):
    T, N, _ = matrices.shape
    tracked_eigvals = np.zeros((T, N), dtype=np.complex128)
    tracked_eigvecs = np.zeros((T, N, N), dtype=np.complex128)
    
    # --- Paso t=0 ---
    eigvals_initial, eigvecs_initial = np.linalg.eig(matrices[0])
    sorted_indices = np.argsort(eigvals_initial.real)
    tracked_eigvals[0] = eigvals_initial[sorted_indices]
    tracked_eigvecs[:, 0] = eigvecs_initial[sorted_indices]
    
    if T == 1:
        return tracked_eigvals, tracked_eigvecs

    # --- Paso t=1 (Aún no tenemos velocidad para predecir) ---
    curr_eigs, curr_eigvecs = np.linalg.eig(matrices[1])
    cost_matrix = np.abs(tracked_eigvals[0][:, None] - curr_eigs[None, :])
    _, col_ind = linear_sum_assignment(cost_matrix)
    tracked_eigvals[1] = curr_eigs[col_ind]
    tracked_eigvecs[:, 1] = curr_eigvecs[col_ind]

    # --- Pasos t > 1 (Uso de extrapolación predictiva) ---
    for t in range(2, T):
        curr_eigs, curr_eigvecs = np.linalg.eig(matrices[t])
        
        # 1. Calculamos la "velocidad" (cuánto se movieron en el último paso)
        velocidad = tracked_eigvals[t-1] - tracked_eigvals[t-2]
        
        # 2. Predecimos dónde deberían estar ahora
        prediccion = tracked_eigvals[t-1] + velocidad
        
        # 3. Emparejamos los actuales con la PREDICCIÓN, no con la posición anterior
        cost_matrix = np.abs(prediccion[:, None] - curr_eigs[None, :])
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        # Guardamos el resultado ordenado
        tracked_eigvals[t] = curr_eigs[col_ind]
        tracked_eigvecs[:, t] = curr_eigvecs[col_ind]
        
    return tracked_eigvals, tracked_eigvecs

def track_eigenvalues(matrices):
    T, N, _ = matrices.shape
    tracked_eigvals = np.zeros((T, N), dtype=np.complex128)
    
    # --- Paso t=0 ---
    eigvals_initial = np.linalg.eigvals(matrices[0])
    tracked_eigvals[0] = eigvals_initial[np.argsort(eigvals_initial.real)]
    
    if T == 1:
        return tracked_eigvals

    # --- Paso t=1 (Aún no tenemos velocidad para predecir) ---
    curr_eigs = np.linalg.eigvals(matrices[1])
    cost_matrix = np.abs(tracked_eigvals[0][:, None] - curr_eigs[None, :])
    _, col_ind = linear_sum_assignment(cost_matrix)
    tracked_eigvals[1] = curr_eigs[col_ind]

    # --- Pasos t > 1 (Uso de extrapolación predictiva) ---
    for t in range(2, T):
        curr_eigs = np.linalg.eigvals(matrices[t])
        
        # 1. Calculamos la "velocidad" (cuánto se movieron en el último paso)
        velocidad = tracked_eigvals[t-1] - tracked_eigvals[t-2]
        
        # 2. Predecimos dónde deberían estar ahora
        prediccion = tracked_eigvals[t-1] + velocidad
        
        # 3. Emparejamos los actuales con la PREDICCIÓN, no con la posición anterior
        cost_matrix = np.abs(prediccion[:, None] - curr_eigs[None, :])
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

def ODE_system(t, y, dim, alpha):
    X = y[:dim]
    W = y[dim:].reshape((dim, dim))

    dim = np.shape(X)[0]
    dWdt = alpha * (np.eye(dim) - np.outer(X, X))
    dXdt = W @ X

    return np.concatenate([dXdt, dWdt.flatten()])

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
    X[0] = X_0
    W[0] = W_0

    I = np.eye(dim)
    for t in range(STEPS):
        

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
time = 10000
dt = 0.1
steps = int(time/dt)
t_eval = np.linspace(0, time, steps)
alpha = 0.001
threshold = 4
rho = 4

np.random.seed(3456)
W_0 = np.random.randn(dim, dim)
u = np.random.standard_normal(dim)
u = u / np.linalg.norm(u)

v = np.random.standard_normal(dim)
v = v - np.dot(u, v) * u
v = v / np.linalg.norm(v)

A = rho * (np.outer(u, v) - np.outer(v, u))


X_0 = np.random.randn(dim)

data = anti_hebbian_network_rk4(steps, W_0, X_0, dt, alpha)

X = data[0][int(4000/dt):]
W = data[1][int(4000/dt):]

plt.plot(X)

plt.show()
