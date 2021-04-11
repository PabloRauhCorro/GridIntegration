import sys, os
import numpy as np
from functools import partial
sys.path.append('..')
sys.path.append(os.getcwd())

from Lebedev import LebedevAngularIntegration
from Gauss_Laguerre import GaussLaguerre
from PeldorGrids.distributions import distributions, parameters, Pn

# number of gridpoints with osPDSFit_math.docx naming convention
L = 38
N = 7
M = 230
K = 4608
J = 7

# Points with a weight that falls below this treshold will be deleted
treshold = 1e-4

# Instantiate the grid integration classes
lebedev = LebedevAngularIntegration()
gauss_laguerre = GaussLaguerre()


# generate the L grid
function = distributions['xi_field']
xi_phi_field_points = lebedev.get_points_spherical(L)
xi_phi_field_weights = lebedev.get_weighted_summands(function, L)
print(xi_phi_field_weights)
# disregard points below treshold 

# generate the N grid
function = partial(distributions['r'], parameters["r_mean"], parameters["r_width"])
r_points = gauss_laguerre.get_points(N)
r_weights = gauss_laguerre.get_weighted_summands(function, N)