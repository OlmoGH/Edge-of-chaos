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
def dot_inplace(M, V, out, DIM):
    for i in range(DIM):
        out[i] = 0
        for j in range(DIM):
            out[i] += M[i, j] * V[j]

@njit(fastmath=True)
def ChunkSimulationSTDP(X, W, Y, INPUT_X, INPUT_X_RANGE, ALPHA, TAU, DT, DIM, CHUNK_NUMBER, CHUNK_STEPS, SKIP):
    saved_steps = CHUNK_STEPS // SKIP
    BUFFER_X = np.zeros((saved_steps, DIM), dtype=X.dtype)
    BUFFER_W = np.zeros((saved_steps, DIM, DIM), dtype=W.dtype)
    
    # Pre-alocamos los arrays temporales
    temp_X = np.empty_like(X)
    k1X = np.empty_like(X)
    k2X = np.empty_like(X)
    k3X = np.empty_like(X)
    k4X = np.empty_like(X)
    k1Y = np.empty_like(X)
    k2Y = np.empty_like(X)
    k3Y = np.empty_like(X)
    k4Y = np.empty_like(X)

    for step in range(CHUNK_STEPS):
        buffer_index = step//SKIP
        global_step = CHUNK_NUMBER * CHUNK_STEPS + step

        if step % SKIP == 0: 
            BUFFER_X[buffer_index] = X.copy()
            BUFFER_W[buffer_index] = W.copy()

        # 1. Empleamos RK4 para la evolución de X y de Y
        # Calculamos k1X
        dot_inplace(W, X, k1X, DIM)
        # Calculamos k1Y
        for i in range(DIM):
            k1X[i] -= X[i]
            k1Y[i] = (X[i] - Y[i])/TAU
        
        # Calculamos k2X
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1X[i] * DT
        dot_inplace(W, temp_X, k2X, DIM)
        # Calculamos k2Y
        for i in range(DIM):
            # k2Y = (X - Y_1)/TAU
            # Y_1 = Y + 0.5 * k1Y
            k2X[i] -= temp_X[i]
            k2Y[i] = (temp_X[i] - (Y[i] + 0.5 * k1Y[i] * DT))/TAU

        # Calculamos k3X
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2X[i] * DT
        dot_inplace(W, temp_X, k3X, DIM)
        # Calculamos k3Y
        for i in range(DIM):
            k3X[i] -= temp_X[i]
            k3Y[i] = (temp_X[i] - (Y[i] + 0.5 * k2Y[i] * DT))/TAU


        # Calculamos k4X
        for i in range(DIM): temp_X[i] = X[i] + k3X[i] * DT
        dot_inplace(W, temp_X, k4X, DIM)
        # Calculamos k4X
        for i in range(DIM):
            k4X[i] -= temp_X[i]
            k4Y[i] = (temp_X[i] - (Y[i] + k3Y[i] * DT))/TAU

        # 2. Actualizamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] += DT * ALPHA * (-(X[i] - Y[i]) * X[j] + X[i] * Y[j] - X[j] * Y[i])
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W[i, i] += DT * ALPHA

        # 3. Actualizamos X e Y y añadimos el input de X
        if INPUT_X_RANGE[0] <= global_step < INPUT_X_RANGE[-1]:
            for i in range(DIM):
                X[i] += DT * (k1X[i] + 2 * k2X[i] + 2 * k3X[i] + k4X[i]) / 6.0 + DT * INPUT_X[global_step - INPUT_X_RANGE[0], i]
                Y[i] += DT * (k1Y[i] + 2 * k2Y[i] + 2 * k3Y[i] + k4Y[i]) / 6.0 
        else:
            for i in range(DIM):
                X[i] += DT * (k1X[i] + 2 * k2X[i] + 2 * k3X[i] + k4X[i]) / 6.0           
                Y[i] += DT * (k1Y[i] + 2 * k2Y[i] + 2 * k3Y[i] + k4Y[i]) / 6.0           



    return X, W, BUFFER_X, BUFFER_W

@njit(fastmath=True)
def StartSimulationSTDP(X, W, ALPHA, TAU, DIM, DT, START):
    
    # Pre-alocamos los arrays temporales
    Y = np.zeros_like(X)
    temp_X = np.empty_like(X)
    k1X = np.empty_like(X)
    k2X = np.empty_like(X)
    k3X = np.empty_like(X)
    k4X = np.empty_like(X)
    k1Y = np.empty_like(X)
    k2Y = np.empty_like(X)
    k3Y = np.empty_like(X)
    k4Y = np.empty_like(X)

    for step in range(START):
        if step % 100_000 == 0: 
            print(f"Simulando paso {step} de {START}")
            print("Paso:", step, "| Max abs(X):", X.max(), "| Max abs(W):", W.max())

        # 1. Empleamos RK4 para la evolución de X y de Y
        # Calculamos k1X
        dot_inplace(W, X, k1X, DIM)
        # Calculamos k1Y
        for i in range(DIM):
            k1X[i] -= X[i]
            k1Y[i] = (X[i] - Y[i])/TAU
        
        # Calculamos k2X
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1X[i] * DT
        dot_inplace(W, temp_X, k2X, DIM)
        # Calculamos k2Y
        for i in range(DIM):
            # k2Y = (X - Y_1)/TAU
            # Y_1 = Y + 0.5 * k1Y
            k2X[i] -= temp_X[i]
            k2Y[i] = (temp_X[i] - (Y[i] + 0.5 * k1Y[i] * DT))/TAU

        # Calculamos k3X
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2X[i] * DT
        dot_inplace(W, temp_X, k3X, DIM)
        # Calculamos k3Y
        for i in range(DIM):
            k3X[i] -= temp_X[i]
            k3Y[i] = (temp_X[i] - (Y[i] + 0.5 * k2Y[i] * DT))/TAU


        # Calculamos k4X
        for i in range(DIM): temp_X[i] = X[i] + k3X[i] * DT
        dot_inplace(W, temp_X, k4X, DIM)
        # Calculamos k4X
        for i in range(DIM):
            k4X[i] -= temp_X[i]
            k4Y[i] = (temp_X[i] - (Y[i] + k3Y[i] * DT))/TAU

        # 2. Actualizamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] += DT * ALPHA * (-(X[i] - Y[i]) * X[j] + X[i] * Y[j] - X[j] * Y[i])
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W[i, i] += DT * ALPHA

        # 3. Actualizamos X e Y
        for i in range(DIM):
            X[i] += DT * (k1X[i] + 2 * k2X[i] + 2 * k3X[i] + k4X[i]) / 6.0
            Y[i] += DT * (k1Y[i] + 2 * k2Y[i] + 2 * k3Y[i] + k4Y[i]) / 6.0

    return X, W, Y

def Simulate_and_save_STDP(X, W, Y, INPUT_X, INPUT_X_RANGE, ALPHA, TAU, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, calc_eigenvalues=False):

    # Creamos o sobreescribimos el archivo hdf5 donde vamos a guardar los datos
    directorio_script = Path(__file__).parent

    ruta_archivo = directorio_script / "Simulacion.h5"

    with h5py.File(ruta_archivo, "w") as f:
        f.attrs["ALPHA"] = ALPHA
        f.attrs["DT"] = DT
        f.attrs["DIM"] = DIM
        f.attrs["SAVED_STEPS"] = (SIMULATED_STEPS) // SKIP
        f.attrs["SKIP"] = SKIP

        # Creamos los datasets que albergan X y W
        dataset_X = f.create_dataset("activity", 
                                    shape=(SIMULATED_STEPS//SKIP, DIM), 
                                    dtype=np.float32,
                                    chunks=True)  
        
        dataset_W = f.create_dataset("connections", 
                                    shape=(SIMULATED_STEPS//SKIP, DIM, DIM), 
                                    dtype=np.float32,
                                    chunks=True,
                                    compression='lzf')
        
        dataset_real_eigvals = f.create_dataset_like("real eigenvalues", dataset_X)
        dataset_imag_eigvals = f.create_dataset_like("imaginary eigenvalues", dataset_X)
        
        # Calculamos el número de lotes que vamos a escribir 
        N_LOTES = SIMULATED_STEPS//CHUNK_STEPS 

        # Índice por el que se empieza a escribir en el dataset
        initial_index = 0
        last_sorted = None
        for i in range(N_LOTES):
            final_index = initial_index + CHUNK_STEPS//SKIP

            # Llamamos a la función para que haga la simulación del lote
            X, W, BUFFER_X, BUFFER_W = ChunkSimulationSTDP(X, W, Y, INPUT_X, INPUT_X_RANGE, ALPHA, TAU, DT, DIM, i, CHUNK_STEPS, SKIP)

            # Guardamos los buffer en el dataset
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