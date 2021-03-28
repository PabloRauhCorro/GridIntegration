import numpy as np
from scipy.special import i0

def von_mises_distr(x, mean, std):
    mu = mean
    kappa = 1/std**2
    return 1/(2*np.pi*i0(kappa)) * np.exp(kappa*np.cos(x-mu))

deg2rad = np.pi / 180.0
