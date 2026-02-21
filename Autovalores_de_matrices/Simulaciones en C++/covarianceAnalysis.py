import numpy as np
import matplotlib.pyplot as plt


alpha = 0.01
dim = 5000
dt = 0.01

raw_data = np.fromfile(f'output/States_{alpha}_{dim}_{dt}.bin', dtype=np.float64)
data = raw_data.reshape((-1, dim))
print(np.shape(data))
init = 30000
covariance = np.cov(data[init:, :].T)
eigenvalues = np.sort(np.linalg.eigvals(covariance))[::-1]
plt.loglog(eigenvalues)
plt.show()

# # Crear coordenadas
# x_pos, y_pos = np.meshgrid(np.arange(dim), np.arange(dim))
# x_pos = x_pos.flatten()
# y_pos = y_pos.flatten()
# z_pos = np.zeros(dim * dim)  # Bases en z=0

# # Dimensiones de las barras
# dx = dy = 0.8  # Ancho de las barras
# dz = (covariance).flatten()  # Alturas

# # Crear figura
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Crear colores según la altura
# colors = plt.cm.viridis(dz / dz.max())

# # Graficar barras
# ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, 
#          color=colors, 
#          alpha=0.8,
#          edgecolor='black',
#          linewidth=0.5)

# ax.set_xlabel('Eje X')
# ax.set_ylabel('Eje Y')
# ax.set_zlabel('Altura')
# ax.set_title('Matriz 3D - Barras Individuales')

# plt.show()