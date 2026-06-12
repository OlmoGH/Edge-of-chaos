import numpy as np
import matplotlib.pyplot as plt
from numba import njit  

@njit(fastmath=True)
def inicializar_red(L):
    """Genera una red aleatoria con espines +1 y -1 compatible con Numba."""
    red = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            red[i, j] = 1 if np.random.rand() > 0.5 else -1
    return red

@njit(fastmath=True)
def paso_metropolis(red, beta):
    """Ejecuta un paso completo de Monte Carlo."""
    L = red.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = red[i, j]
        
        vecinos = (red[(i + 1) % L, j] +
                   red[(i - 1) % L, j] +
                   red[i, (j + 1) % L] +
                   red[i, (j - 1) % L])
        
        dE = 2 * s * vecinos
        
        if dE <= 0 or np.random.rand() < np.exp(-dE * beta):
            red[i, j] = -s
    return red

@njit(fastmath=True)
def simular_ising(L, T, pasos_mcs):
    """Simula el modelo de Ising para una temperatura dada."""
    beta = 1.0 / T
    red = inicializar_red(L)
    for paso in range(pasos_mcs):
        red = paso_metropolis(red, beta)
    return red

# Definición de parámetros
L = 4000       
MCS_LOW = 4000    
MCS_HIGH = 500    
T_LOW = 2       
T_HIGH = 4.0    
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))

print("Simulando temperatura crítica (T = 2.269)...")
red_critica = simular_ising(L, T_C, MCS_LOW)

plt.imshow(red_critica, cmap='coolwarm', interpolation='None')
plt.rcParams.update({'font.size': 13, 'font.family': 'serif'})
plt.title("Punto crítico\n T ~ 2.269")
plt.xticks([])
plt.yticks([])
plt.savefig("Ising_critico_sin_labels.pdf")
plt.savefig("Ising_critico_sin_labels.png", dpi=500)
plt.show()
# # --- EJECUCIÓN DE LAS SIMULACIONES ---
# print("Simulando temperatura baja (T = 1.5)...")
# red_baja = simular_ising(L, T_LOW, MCS_LOW)

# print("Simulando temperatura alta (T = 4.0)...")
# red_alta = simular_ising(L, T_HIGH, MCS_HIGH)

# # --- GENERACIÓN DE LAS GRÁFICAS PARA EL TFG ---
# plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# # Creamos la figura con un tamaño fijo
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

# # Ajustamos los márgenes manualmente para dejar espacio libre ABAJO (bottom=0.22)
# fig.subplots_adjust(left=0.08, right=0.92, bottom=0.2, top=0.85, )

# # Gráfica 1: Temperatura Baja
# im1 = ax1.imshow(red_baja, cmap='coolwarm', interpolation='nearest')
# ax1.set_title(f"Fase Ordenada (Baja Temperatura)\nT = {T_LOW}", fontsize=13, pad=12)
# ax1.set_xticks([])
# ax1.set_yticks([])

# # Gráfica 2: Temperatura Alta
# im2 = ax2.imshow(red_alta, cmap='coolwarm', interpolation='nearest')
# ax2.set_title(f"Fase Desordenada (Alta Temperatura)\nT = {T_HIGH}", fontsize=13, pad=12)
# ax2.set_xticks([])
# ax2.set_yticks([])

# # CREACIÓN DEL EJE PROPIO PARA LA COLORBAR (cax)
# # Coordenadas: [izquierda, abajo, ancho, alto] en porcentaje de la figura
# # 0.3 significa que empieza al 30% del ancho; 0.08 significa que está al 8% desde el borde inferior.
# cax = fig.add_axes([0.30, 0.1, 0.40, 0.04])

# # Dibujamos la barra en ese eje específico
# cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
# cbar.set_ticks([-1, 1])
# cbar.set_ticklabels(['Espín Abajo (-1)', 'Espín A (+1)'])

# # Guardar la imagen con alta calidad
# nombre_archivo = 'transicion_fase_ising_corregida.png'
# plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
# print(f"\n¡Gráfica corregida guardada como '{nombre_archivo}'!")

# plt.show()