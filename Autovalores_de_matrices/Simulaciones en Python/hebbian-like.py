import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from eigenshuffle import eigenshuffle_eig
import numpy as np
from scipy.optimize import linear_sum_assignment

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

def ODE_system(t, y, dim, alpha):
    X = y[:dim]
    W = y[dim:].reshape((dim, dim))

    dim = np.shape(X)[0]
    dWdt = alpha * (np.eye(dim) - np.outer(X, X))
    dXdt = W @ X

    return np.concatenate([dXdt, dWdt.flatten()])


dim = 10
time = 50000
t_eval = np.linspace(10000, 20000, 20000)
alpha = 0.0001

rng_X = np.random.default_rng(42)
rng_W = np.random.default_rng(69)

X_0 = rng_X.normal(loc=0, scale=1/np.sqrt(dim), size=dim)
W_0 = rng_W.normal(loc=0, scale=1/np.sqrt(dim), size=(dim, dim))

y_0 = np.concatenate([X_0, W_0.flatten()])
solution = solve_ivp(ODE_system, [0, time], y0=y_0, args=[dim, alpha], t_eval=t_eval)

X = solution.y[:dim, :].transpose()
W = solution.y[dim:, :].reshape((dim, dim, -1)).transpose(2, 0, 1)

W_norm = np.linalg.norm(W, axis=(1, 2))
X_norm = np.linalg.norm(X, axis=1)

y = X_norm**2
y_dotdot = np.gradient(np.gradient(y, t_eval), t_eval)

plt.plot(y_dotdot, y, 'b.', ms=1)
plt.show()

V = 0.5 * W_norm**2 + alpha * (0.5 * X_norm**2 - np.log(X_norm))
dVdt = (np.trace(W, axis1=1, axis2=2) - np.einsum('ti, tij, tj -> t', X, W, X)/X_norm**2)
V_dot = np.gradient(V, t_eval)

plt.plot(t_eval, (V - V.mean())/(V.max()-V.min()), label='Lyapunov function')
plt.plot(t_eval, (V_dot - V_dot.mean())/(V_dot.max()-V_dot.min()), label=r"$\frac{dV}{dt}$ numérica", alpha=0.8)
plt.plot(t_eval, (dVdt - dVdt.mean())/(dVdt.max()-dVdt.min()), label=r"$\frac{dV}{dt}$ analítica", alpha=0.8)
plt.legend()
plt.show()







# print("X: ", X.shape)
# print("W: ", W.shape)
# WX = np.einsum('tij, tj -> ti', W, X)
# WX_norm = np.linalg.norm(WX, axis=1)
# X_norm = np.linalg.norm(X, axis=1)

# W_spectral_norm = np.linalg.norm(W, ord=2, axis=(1, 2))

# plt.plot(WX_norm, label=r"$|WX|$")
# plt.plot(W_norm * X_norm/2, label=r"$|W||X|$")
# # plt.plot(W_spectral_norm * X_norm, label=r"$\sigma_{max}(W)|X|$")
# plt.legend()
# plt.show()

