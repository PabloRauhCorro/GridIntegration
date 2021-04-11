
import numpy as np
from functools import partial
from supplementary.von_mises_distribution import von_mises_distr

# test distributions with test parameters for prototyping


def Pn(x, mean, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mean)**2 / (2*sigma**2))


parameters = {}
parameters["r_mean"] = 1.94
parameters["r_width"] = 0.03
parameters["xi_mean"] = 90.0
parameters["xi_width"] = 10.0
parameters["phi_mean"] = 180
parameters["phi_width"] = 5.0
parameters["alpha_mean"] = 180
parameters["alpha_width"] = 20.0
parameters["beta_mean"] = 45.0
parameters["beta_width"] = 20.0
parameters["gamma_mean"] = 0.0
parameters["gamma_width"] = 20.0
parameters["rel_prob"] = 1.0
parameters["j_mean"] = 0.0
parameters["j_width"] = 0.0


distributions = {}
distributions['xi_field'] = lambda xi, phi: np.sin(xi)
distributions['r'] = Pn
distributions['xi'] = von_mises_distr
distributions['phi'] = von_mises_distr
distributions['alpha'] = von_mises_distr
distributions['beta'] = von_mises_distr
distributions['gamma'] = von_mises_distr
distributions['J'] = Pn

