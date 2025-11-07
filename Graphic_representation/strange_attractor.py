import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

a = 0.95
b = 0.7
c = 0.6
d = 3.5
e = 0.25
f = 0.1

def system(x, y, z):
    global a, b, c, d, e, f
    dxdt = (z-b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - z ** 3 / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * x ** 3
    return dxdt, dydt, dzdt

initial_condition = np.array([[1, 1.001], [0, 0], [0, 0]])
n_steps = 100000
dt = 0.01
states = np.empty((n_steps, 3, 2), float)
states[0] = initial_condition

for i in range(n_steps-1):
    x = states[i, 0, :]
    y = states[i, 1, :]
    z = states[i, 2, :]

    system_values = system(x, y, z)
    dxdt = system_values[0]
    dydt = system_values[1]
    dzdt = system_values[2]

    x += dt * dxdt
    y += dt * dydt
    z += dt * dzdt

    states[i+1] = [x, y, z]

axes = plt.axes(projection='3d')

axes.plot3D(states[:, 0, 0], states[:, 1, 0], states[:, 2, 0], 'b-', markersize=1)
axes.plot3D(states[:, 0, 1], states[:, 1, 1], states[:, 2, 1], 'r-', markersize=1)


plt.show()
# Coeficientes de Lyapunov
# r = (states[:, :, 0]-states[:, :, 1]) ** 2
# r = r.sum(axis=1)
# axes2 = plt.axes()
# axes2.set_yscale('linear')
# axes2.plot(r)
# plt.show()
