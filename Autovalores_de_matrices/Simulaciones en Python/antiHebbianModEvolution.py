import numpy as np
from numba import njit
from pathlib import Path
import h5py

@njit(fastmath=True)
def X_dot(W, X, DIM):
    phi = np.zeros_like(X)

    # Aplicamos la función sigmoide
    for i in range(DIM):
        phi[i] = np.tanh(X[i])
    
    result = np.zeros_like(X)
    for i in range(DIM):
        for j in range(DIM):
            result[i] += W[i, j] * phi[j]
    
    return result

@njit(fastmath=True)
def SigmoidChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP):
    saved_steps = CHUNK_STEPS // SKIP
    BUFFER_X = np.zeros((saved_steps, DIM), dtype=X.dtype)
    BUFFER_W = np.zeros((saved_steps, DIM, DIM), dtype=W.dtype)
    
    # Pre-alocamos los arrays temporales
    temp_X = np.zeros(DIM, dtype=X.dtype)
    W_dot = np.zeros_like(W)

    for step in range(CHUNK_STEPS):
        buffer_index = step//SKIP

        if step % SKIP == 0: 
            BUFFER_X[buffer_index] = X.copy()
            BUFFER_W[buffer_index] = W.copy()

        # 1. Empleamos RK4 para la evolución de x
        # Calculamos k1
        k1 = X_dot(W, X, DIM)
        
        # Calculamos k2
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1[i] * DT
        
        k2 = X_dot(W, temp_X, DIM)

        # Calculamos k3
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2[i] * DT
        k3 = X_dot(W, temp_X, DIM)

        # Calculamos k4
        for i in range(DIM): temp_X[i] = X[i] + k3[i] * DT
        k4 = X_dot(W, temp_X, DIM)

        # Calculamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W_dot[i, j] = -DT * ALPHA * X[i] * X[j]
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W_dot[i, i] += ALPHA

        # # Multiplicamos por W
        # for i in range(DIM):
        #     for j in range(DIM):
        #         sum = 0
        #         for k in range(DIM):
        #             sum += W_dot[i, k] * W[k, j]
        #         W_dot[i, j] = sum

        # Actualizamos X
        for i in range(DIM):
            X[i] += DT * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0

        # Actualizamos W
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] += DT * W_dot[i, j]

    return X, W, BUFFER_X, BUFFER_W

@njit(fastmath=True)
def StartSimulation(X, W, ALPHA, DIM, DT, START):
    
    # Pre-alocamos los arrays temporales
    temp_X = np.zeros(DIM, dtype=X.dtype)
    W_dot = np.zeros_like(W)


    for step in range(START):
        if step % 1_000_000 == 0: 
            print(f"Simulando paso {step} de {START}")
            print("Paso:", step, "| Max abs(X):", X.max(), "| Max abs(W):", W.max())
        # 1. Empleamos RK4 para la evolución de x
        # Calculamos k1
        k1 = X_dot(W, X, DIM)
        
        # Calculamos k2
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1[i] * DT
        
        k2 = X_dot(W, temp_X, DIM)

        # Calculamos k3
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2[i] * DT
        k3 = X_dot(W, temp_X, DIM)

        # Calculamos k4
        for i in range(DIM): temp_X[i] = X[i] + k3[i] * DT
        k4 = X_dot(W, temp_X, DIM)

        # Calculamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W_dot[i, j] = -DT * ALPHA * X[i] * X[j]
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W_dot[i, i] += ALPHA

        # # Multiplicamos por W
        # for i in range(DIM):
        #     for j in range(DIM):
        #         sum = 0
        #         for k in range(DIM):
        #             sum += W_dot[i, k] * W[k, j]
        #         W_dot[i, j] = sum

        # Actualizamos X
        for i in range(DIM):
            X[i] += DT * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0

        # Actualizamos W
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] += DT * W_dot[i, j]
    return X, W

def Simulate_and_save(ALPHA, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, START, calc_eigenvalues=False):

    W = np.random.normal(0, 1.0/np.sqrt(DIM), (DIM, DIM))

    X = np.random.normal(0, 1.0, DIM)

    directorio_script = Path(__file__).parent

    ruta_archivo = directorio_script / "Simulacion.h5"

    with h5py.File(ruta_archivo, "w") as f:
        f.attrs["ALPHA"] = ALPHA
        f.attrs["DT"] = DT
        f.attrs["DIM"] = DIM
        f.attrs["SAVED_STEPS"] = (SIMULATED_STEPS - START) // SKIP
        f.attrs["SKIP"] = SKIP

        # Creamos los datasets que albergan X y W
        dataset_X = f.create_dataset("activity", 
                                    shape=((SIMULATED_STEPS - START)//SKIP, DIM), 
                                    dtype=np.float32,
                                    chunks=(1, DIM))  
        
        dataset_W = f.create_dataset("connections", 
                                    shape=((SIMULATED_STEPS - START)//SKIP, DIM, DIM), 
                                    dtype=np.float32,
                                    chunks=(1, DIM, DIM),
                                    compression='lzf')
        
        dataset_real_eigvals = f.create_dataset_like("real eigenvalues", dataset_X)
        dataset_imag_eigvals = f.create_dataset_like("imaginary eigenvalues", dataset_X)
        
        # Calculamos el número de lotes que vamos a escribir 
        N_LOTES = (SIMULATED_STEPS - START)//CHUNK_STEPS 

        # Índice por el que se empieza a escribir en el dataset
        initial_index = 0

        # Simulamos los primeros START pasos
        print(f"Simulando los primeros {START} pasos")
        X, W = StartSimulation(X, W, ALPHA, DIM, DT, START)

        for i in range(N_LOTES):
            final_index = initial_index + CHUNK_STEPS//SKIP

            # Llamamos a la función para que haga la simulación del lote
            X, W, BUFFER_X, BUFFER_W = SigmoidChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP)

            # Guardamos los buffer en el dataset
            dataset_X[initial_index:final_index] = BUFFER_X
            dataset_W[initial_index:final_index] = BUFFER_W

            if calc_eigenvalues:
                chunk_eigvals = np.linalg.eigvals(BUFFER_W)
                dataset_real_eigvals[initial_index:final_index] = chunk_eigvals.real
                dataset_imag_eigvals[initial_index:final_index] = chunk_eigvals.imag

            f.flush()

            initial_index = final_index
            print(f"Lote {i+1}/{N_LOTES} completado y guardado en disco")

    print("Simulación completada")