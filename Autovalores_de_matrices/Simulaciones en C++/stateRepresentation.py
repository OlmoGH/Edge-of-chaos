import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

states = pd.read_csv("States.txt", sep=",", header=0).to_numpy()

print(np.shape(states))
mask = states < 1
states[mask] = 0
plt.imshow(states.T, aspect='auto', cmap='hot', interpolation='none')
plt.show()
