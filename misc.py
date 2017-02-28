
import numpy as np

def init_weight(num_in, num_out):
    max_wt = np.sqrt(6.0 / (num_in + num_out)) * 2
    return (np.random.rand(num_in, num_out) - 0.5) * max_wt

def sigmoid(inp):
    return np.exp(-np.logaddexp(0, -inp))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
