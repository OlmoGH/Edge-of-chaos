import numpy as np
import h5py
from numba import njit
from pathlib import Path

def read_data():
    """
    Lee los datos de la simulación desde un archivo HDF5 y, opcionalmente, calcula sus autovalores.

    La función busca el archivo 'Simulacion.h5' en el mismo directorio que el script. 
    Carga en memoria los datasets de actividad ('activity') y conexiones ('connections'). 
    Si se especifica, procesa las matrices de conexiones en lotes para calcular sus 
    autovalores, optimizando el uso de memoria RAM, y guarda los resultados de vuelta 
    en el propio archivo HDF5.

    Args:
        chunk_size (int, opcional): Tamaño de los lotes en los que se procesa el dataset 
            de la matriz de conexiones para calcular los autovalores, 10_000 por defecto.

    Returns:
        tuple: Una tupla con cuatro elementos:
            - X (numpy.ndarray): Datos extraídos del dataset 'activity'.
            - W (numpy.ndarray): Datos extraídos del dataset 'connections'.
            - real_eigvals_array (numpy.ndarray): Parte real de los autovalores 
              (o una matriz de ceros si read_eigenvalues es False).
            - imag_eigvals_array (numpy.ndarray): Parte imaginaria de los autovalores 
              (o una matriz de ceros si read_eigenvalues es False).

    Note:
        Esta función requiere que la función auxiliar `obtener_autovalores_lote()` 
        esté definida en el mismo contexto. Además, modifica el archivo original 
        ('Simulacion.h5') si `read_eigenvalues` es True.
    """

    # Directorio y ruta del archivo .h5
    directorio_script = Path(__file__).parent
    ruta_archivo = directorio_script / "Simulacion.h5"

    # Si no sólo calculamos leemos los datos

    f = h5py.File(ruta_archivo, "r")
    dt = f.attrs["DT"]
    skip = f.attrs["SKIP"]
    dim = f.attrs["DIM"]
    alpha = f.attrs["ALPHA"]
    steps = f.attrs["SAVED_STEPS"]
    dataset_X = f["activity"]
    dataset_W = f["connections"]
    dataset_real_eigvals = f["real eigenvalues"]
    dataset_imag_eigvals = f["imaginary eigenvalues"]

            
    return f, dataset_X, dataset_W, dataset_real_eigvals, dataset_imag_eigvals