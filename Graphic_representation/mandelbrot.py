import numpy as np
import matplotlib.pyplot as plt

array_sizex = 3000
array_sizey = 2000
real = np.linspace(-2, 1, array_sizex)
imag = 1j * np.linspace(-1, 1, array_sizey)
plane = np.sum(np.meshgrid(imag, real), axis=0)
figure = np.zeros_like(plane, dtype=int)
print(np.shape(plane))

for i in range(array_sizex):
    for j in range(array_sizey):
        z = 0
        c = plane[i, j]
        for a in range(100):
            if np.abs(z) > 2:
                figure[i, j] = 1
                break
            z = np.pow(z, 2) + c

plt.imshow(figure, cmap='grey')
plt.show()