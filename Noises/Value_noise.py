import numpy as np
import matplotlib.pyplot as plt

def lerp(start, end, t):
    return start + (end - start) * t

def smoothstep(start, end, t):
    return start + (end - start) * t * t * (3 - 2 * t)

def generate_value_noise(grid_size, scale, use_smoothing=True):
    edges = np.random.uniform(-1, 1, (scale+1, scale+1))
    split = grid_size / scale
    h_out = grid_size
    w_out = grid_size

    y_indices, x_indices = np.ogrid[:h_out, :w_out]

    grid_x = (x_indices / split).astype(int)
    grid_y = (y_indices / split).astype(int)

    t_x = (x_indices % split) / split
    t_y = (y_indices % split) / split

    top_left = edges[grid_y, grid_x]
    top_right = edges[grid_y, grid_x + 1]
    bottom_left = edges[grid_y + 1, grid_x]
    bottom_right = edges[grid_y + 1, grid_x + 1]

    if use_smoothing:
        top_lerp = smoothstep(top_left, top_right, t_x)
        bottom_lerp = smoothstep(bottom_left, bottom_right, t_x)
        values = smoothstep(top_lerp, bottom_lerp, t_y)
    else:
        top_lerp = lerp(top_left, top_right, t_x)
        bottom_lerp = lerp(bottom_left, bottom_right, t_x)
        values = lerp(top_lerp, bottom_lerp, t_y)

    return values

steps = 10
grid_size = 2**steps

fractal_noise = np.zeros((grid_size, grid_size))

for i in range(1, steps):
    fractal_noise += generate_value_noise(grid_size, 2 ** i) / (2 ** i)

plt.imshow(fractal_noise, cmap='gray')

plt.show()