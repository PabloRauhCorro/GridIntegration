import numpy as np
from scipy.special import i0

def von_mises_distr(x, mean, width):
    kappa =  1 / width**2
    if np.isfinite(i0(kappa)):
        return np.exp(kappa * np.cos(x - mean)) / (2*np.pi * i0(kappa))
    else:
        return np.where(x == mean, 1.0, 0.0)

deg2rad = np.pi / 180.0

def uniform_distr(x, mean, width):
    return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0)