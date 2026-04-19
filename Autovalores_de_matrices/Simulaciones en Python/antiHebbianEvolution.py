import numpy as np
from numba import njit
from pathlib import Path
import h5py

@njit(fastmath=True)
def ChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP):
    saved_steps = CHUNK_STEPS // SKIP
    BUFFER_X = np.zeros((saved_steps, DIM), dtype=X.dtype)
    BUFFER_W = np.zeros((saved_steps, DIM, DIM), dtype=W.dtype)
    
    # Pre-alocamos los arrays temporales
    temp_X = np.zeros(DIM, dtype=X.dtype)

    for step in range(CHUNK_STEPS):
        buffer_index = step//SKIP

        if step % SKIP == 0: 
            BUFFER_X[buffer_index] = X.copy()
            BUFFER_W[buffer_index] = W.copy()

        # 1. Empleamos RK4 para la evolución de x
        # Calculamos k1
        k1 = np.dot(W, X)
        
        # Calculamos k2
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1[i] * DT
        
        k2 = np.dot(W, temp_X)

        # Calculamos k3
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2[i] * DT
        k3 = np.dot(W, temp_X)

        # Calculamos k4
        for i in range(DIM): temp_X[i] = X[i] + k3[i] * DT
        k4 = np.dot(W, temp_X)

        # 2. Actualizamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] -= DT * ALPHA * X[i] * X[j]
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W[i, i] += DT * ALPHA

        # 3. Actualizamos X
        for i in range(DIM):
            X[i] += DT * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0

    return X, W, BUFFER_X, BUFFER_W

@njit(fastmath=True)
def StartSimulation(X, W, ALPHA, DIM, DT, START):
    
    # Pre-alocamos los arrays temporales
    temp_X = np.zeros(DIM, dtype=X.dtype)
    k1 = np.zeros(DIM, dtype=X.dtype)
    k2 = np.zeros(DIM, dtype=X.dtype)
    k3 = np.zeros(DIM, dtype=X.dtype)
    k4 = np.zeros(DIM, dtype=X.dtype)

    for step in range(START):
        if step % 10_000 == 0: 
            print(f"Simulando paso {step} de {START}")
            print("Paso:", step, "| Max abs(X):", X.max(), "| Max abs(W):", W.max())
        # 1. Empleamos RK4 para la evolución de x
        # Calculamos k1
        for i in range(DIM):
            k1[i] = 0.0
            for j in range(DIM):
                k1[i] += W[i, j] * X[j]
        
        # Calculamos k2
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k1[i] * DT
        
        for i in range(DIM):
            k2[i] = 0.0
            for j in range(DIM):
                k2[i] += W[i, j] * temp_X[j]

        # Calculamos k3
        for i in range(DIM): temp_X[i] = X[i] + 0.5 * k2[i] * DT

        for i in range(DIM):
            k3[i] = 0.0
            for j in range(DIM):
                k3[i] += W[i, j] * temp_X[j]

        # Calculamos k4
        for i in range(DIM): temp_X[i] = X[i] + k3[i] * DT

        for i in range(DIM):
            k4[i] = 0.0
            for j in range(DIM):
                k4[i] += W[i, j] * temp_X[j]

        # 2. Actualizamos el término de xx^T
        for i in range(DIM):
            for j in range(DIM):
                W[i, j] -= DT * ALPHA * X[i] * X[j]
        
        # Actualizamos la diagonal con la identidad
        for i in range(DIM):
            W[i, i] += DT * ALPHA

        # 3. Actualizamos X
        for i in range(DIM):
            X[i] += DT * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0

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
            X, W, BUFFER_X, BUFFER_W = ChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP)

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