import numpy as np
import matplotlib.pyplot as plt
import gc

def get_data(alpha, dim, dt):
    alpha_str = np.format_float_positional(alpha, trim='-')
    raw_data = np.fromfile(f'output/States_{alpha_str}_{dim}_{dt}.bin', dtype=np.float64)
    states = raw_data.reshape((-1, dim)).T
    return states

def save_heatmap(states, alpha, n_sigmas=0):
    mu = states.mean()
    sigma = states.std()
    mask = states < mu + n_sigmas * sigma
    states[mask] = 0
    plt.imshow(states, aspect='auto', cmap='hot', interpolation='none')
    plt.colorbar()
    plt.savefig(f"../Animaciones y figuras/heatmap_{alpha}.png")

def show_hist(states):
    plt.hist(states.ravel(), 100, density=True)
    plt.show()

def show_fft(states, min_val=-np.inf):
    mask = states < min_val
    states[mask] = 0
    states_resumado = states.sum(axis=0)
    plt.plot(np.abs(np.fft.fft(states_resumado))[1:len(states_resumado)//2])
    plt.show()

alpha = 0.01
dim = 300
dt = 0.01
states = get_data(alpha, dim, dt)
save_heatmap(states, alpha, 2)