import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01
dim = 300
dt = 0.01

# Lectura de los datos y conversión autovalores = [[re1, im1], [re2, im2], ...]
raw_data = np.fromfile(f'output/Delta_{alpha}_{dim}_{dt}.bin')
data = raw_data.resize(-1, dim)
data = (data.T / data[:, 0])
data = np.log(np.abs(data))
delta = np.mean(data, axis=1)
plt.plot(delta)

plt.show()
