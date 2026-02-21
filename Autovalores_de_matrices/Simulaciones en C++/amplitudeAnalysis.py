import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación necesarios para localizzar el archivo de datos
alpha = 0.01
dim = 2
dt = 0.01

# Lectura de los datos y conversión autovalores = [[re1, im1], [re2, im2], ...]
data = np.loadtxt(f'output/Eigenvalues_{alpha:.2f}_{dim}_{dt}.txt')
real = data[:dim]
imag = data[dim:]

plt.plot(real.max(axis=1) - real.min(axis=1))


max = real.max(axis=1).mean()
min = real.min(axis=1).mean()

A_mean = (real.max(axis=1) - real.min(axis=1)).mean()
A_max = real.max() - real.min()
print(f"La amplitud promedio es {A_mean:.2f}")
print(f"La amplitud máxima es {A_max:.2f}")

plt.show()

print(np.shape(real))