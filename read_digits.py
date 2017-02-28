import numpy as np

def read_digits(filename):
    with open(filename) as txt:
        data = np.loadtxt(txt, delimiter=',')
        x_data, y_data = data[:, :-1], data[:, -1]
        y_data = np.array([int(y) for y in y_data])
        bin_x = 1.0 * (x_data > 0.5)
        bin_xy= (bin_x, y_data)
    return bin_xy
    
