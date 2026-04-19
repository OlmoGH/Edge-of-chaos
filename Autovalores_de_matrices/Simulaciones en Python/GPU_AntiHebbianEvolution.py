import numpy as np
import cupy as cp
import h5py
from pathlib import Path

# ¡Adiós Numba! CuPy prefiere las operaciones vectorizadas puras
def ChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP):
    saved_steps = CHUNK_STEPS // SKIP
    
    # Alocamos los buffers directamente en la GPU (VRAM)
    BUFFER_X = cp.zeros((saved_steps, DIM), dtype=X.dtype)
    BUFFER_W = cp.zeros((saved_steps, DIM, DIM), dtype=W.dtype)
    
    # Precalculamos constantes para no repetirlas en el bucle
    dt_alpha = DT * ALPHA
    I_dt_alpha = cp.eye(DIM, dtype=W.dtype) * dt_alpha
    dt_6 = DT / 6.0

    for step in range(CHUNK_STEPS):
        if step % SKIP == 0: 
            buffer_index = step // SKIP
            BUFFER_X[buffer_index] = X.copy()
            BUFFER_W[buffer_index] = W.copy()

        # 1. RK4 100% Vectorizado (¡Esto vuela en la RTX 5050!)
        k1 = cp.dot(W, X)
        k2 = cp.dot(W, X + 0.5 * DT * k1)
        k3 = cp.dot(W, X + 0.5 * DT * k2)
        k4 = cp.dot(W, X + DT * k3)

        # 2. Actualizamos W usando el producto externo (Outer product)
        # W = W - dt*alpha*(X * X^T) + dt*alpha*Identidad
        W -= dt_alpha * cp.outer(X, X)
        W += I_dt_alpha

        # 3. Actualizamos X
        X += dt_6 * (k1 + 2*k2 + 2*k3 + k4)

    # Convertimos los buffers a NumPy (RAM) justo antes de devolverlos para que h5py pueda guardarlos
    return X, W, cp.asnumpy(BUFFER_X), cp.asnumpy(BUFFER_W)

def StartSimulation(X, W, ALPHA, DIM, DT, START):
    # Precalculamos constantes
    dt_alpha = DT * ALPHA
    I_dt_alpha = cp.eye(DIM, dtype=W.dtype) * dt_alpha
    dt_6 = DT / 6.0

    for step in range(START):
        if step % 10_000 == 0:  # Reducido un poco para que veas que avanza rápido
            print(f"Simulando paso {step} de {START}")
            # Usamos .get() para imprimir valores en la CPU
            print("Paso:", step, "| Max abs(X):", float(cp.max(cp.abs(X))), "| Max abs(W):", float(cp.max(cp.abs(W))))
            
        # 1. RK4 Vectorizado
        k1 = cp.dot(W, X)
        k2 = cp.dot(W, X + 0.5 * DT * k1)
        k3 = cp.dot(W, X + 0.5 * DT * k2)
        k4 = cp.dot(W, X + DT * k3)

        # 2. Actualizamos W
        W -= dt_alpha * cp.outer(X, X)
        W += I_dt_alpha

        # 3. Actualizamos X
        X += dt_6 * (k1 + 2*k2 + 2*k3 + k4)

    return X, W

def Simulate_and_save(ALPHA, DT, DIM, SIMULATED_STEPS, CHUNK_STEPS, SKIP, START, calc_eigenvalues=False):
    # Inicializamos X y W directamente en la GPU usando CuPy
    W = cp.random.normal(0, 1.0/np.sqrt(DIM), (DIM, DIM), dtype=cp.float32)
    X = cp.random.normal(0, 1.0, DIM, dtype=cp.float32)

    directorio_script = Path(__file__).parent
    ruta_archivo = directorio_script / "Simulacion.h5"

    with h5py.File(ruta_archivo, "w") as f:
        f.attrs["ALPHA"] = ALPHA
        f.attrs["DT"] = DT
        f.attrs["DIM"] = DIM
        f.attrs["SAVED_STEPS"] = (SIMULATED_STEPS - START) // SKIP
        f.attrs["SKIP"] = SKIP

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
        
        N_LOTES = (SIMULATED_STEPS - START) // CHUNK_STEPS 
        initial_index = 0

        print(f"Simulando los primeros {START} pasos")
        X, W = StartSimulation(X, W, ALPHA, DIM, DT, START)

        for i in range(N_LOTES):
            final_index = initial_index + CHUNK_STEPS//SKIP

            X, W, BUFFER_X, BUFFER_W = ChunkSimulation(X, W, ALPHA, DIM, DT, CHUNK_STEPS, SKIP)

            # BUFFER_X y BUFFER_W ya son arrays de NumPy aquí, listos para h5py
            dataset_X[initial_index:final_index] = BUFFER_X
            dataset_W[initial_index:final_index] = BUFFER_W

            # El cálculo de autovalores lo seguimos haciendo en CPU por estabilidad 
            # (CuPy a veces da problemas con matrices asimétricas no hermíticas)
            if calc_eigenvalues:
                # Calculamos autovalores de cada matriz del lote
                # np.linalg funciona porque BUFFER_W es NumPy array
                chunk_eigvals = np.linalg.eigvals(BUFFER_W)
                dataset_real_eigvals[initial_index:final_index] = chunk_eigvals.real
                dataset_imag_eigvals[initial_index:final_index] = chunk_eigvals.imag

            f.flush()
            initial_index = final_index
            print(f"Lote {i+1}/{N_LOTES} completado y guardado en disco")

    print("Simulación completada")