import numpy as np
from numba import njit
from pathlib import Path
import h5py
from scipy.optimize import linear_sum_assignment

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

@njit(fastmath=True)
def ChunkSimulationSTDP(X, W, INPUT_X, ALPHA, TAU, DT, DIM, CHUNK_NUMBER, CHUNK_STEPS, SKIP):
    saved_steps = CHUNK_STEPS // SKIP
    BUFFER_X = np.zeros((saved_steps, DIM), dtype=X.dtype)
    BUFFER_W = np.zeros((saved_steps, DIM, DIM), dtype=W.dtype)
    
    I = np.eye(DIM) # Matriz identidad para la actualización de la diagonal
    Y = np.zeros_like(X)

    for step in range(CHUNK_STEPS):
        global_step = CHUNK_NUMBER * CHUNK_STEPS + step

        if step % SKIP == 0: 
            BUFFER_X[step // SKIP] = X.copy()
            BUFFER_W[step // SKIP] = W.copy()

        # 1. Evolución temporal de X e Y (Estrictamente Lineal)
        dX_dt = np.dot(W, X) + INPUT_X[step]
        dY_dt = (X - Y) / TAU

        # 2. Aprendizaje, homeostasis y ruido vectorizado usando X directamente
        noise = np.random.randn(DIM, DIM) / np.sqrt(DIM)
        dW = I - np.outer(X, X) + np.outer(X, Y) - np.outer(Y, X) + noise
        
        # 3. Integración de Euler simple
        X += DT * dX_dt
        Y += DT * dY_dt
        W += DT * ALPHA * dW 

    return X, W, BUFFER_X, BUFFER_W

def Simulate_and_save_STDP(X, W, INPUT_X, ALPHA, TAU, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, calc_eigenvalues=False):
    directorio_script = Path(__file__).parent
    ruta_archivo = directorio_script / "Simulacion.h5"

    with h5py.File(ruta_archivo, "w") as f:
        f.attrs["ALPHA"] = ALPHA
        f.attrs["DT"] = DT
        f.attrs["DIM"] = DIM
        f.attrs["SAVED_STEPS"] = (SIMULATED_STEPS) // SKIP
        f.attrs["SKIP"] = SKIP

        dataset_X = f.create_dataset("activity", shape=(SIMULATED_STEPS//SKIP, DIM), dtype=np.float32, chunks=True)  
        dataset_W = f.create_dataset("connections", shape=(SIMULATED_STEPS//SKIP, DIM, DIM), dtype=np.float32, chunks=True, compression='lzf')
        dataset_real_eigvals = f.create_dataset_like("real eigenvalues", dataset_X)
        dataset_imag_eigvals = f.create_dataset_like("imaginary eigenvalues", dataset_X)
        
        N_LOTES = SIMULATED_STEPS // CHUNK_STEPS 
        initial_index = 0
        last_sorted = None

        for i in range(N_LOTES):
            final_index = initial_index + CHUNK_STEPS // SKIP
            X, W, BUFFER_X, BUFFER_W = ChunkSimulationSTDP(X, W, INPUT_X, ALPHA, TAU, DT, DIM, i, CHUNK_STEPS, SKIP)

            dataset_X[initial_index:final_index] = BUFFER_X
            dataset_W[initial_index:final_index] = BUFFER_W

            if calc_eigenvalues:
                chunk_eigvals = np.linalg.eigvals(BUFFER_W)
                sorted_chunk_eigenvalues = sort_eigenvalues(last_sorted, chunk_eigvals)
                dataset_real_eigvals[initial_index:final_index] = sorted_chunk_eigenvalues.real
                dataset_imag_eigvals[initial_index:final_index] = sorted_chunk_eigenvalues.imag
                last_sorted = sorted_chunk_eigenvalues[-1]

            f.flush()
            initial_index = final_index
            print(f"Lote {i+1}/{N_LOTES} completado y guardado en disco")

    print("Simulación completada")