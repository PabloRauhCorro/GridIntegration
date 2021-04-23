import sys, os
import numpy as np
from scipy import integrate
sys.path.append(os.getcwd())
from Gauss_Laguerre import GaussLaguerre
from supplementary.distributions import von_mises_distr
from supplementary.deg2rad import deg2rad
from tests.num_points_vs_deviation.plot_deviation import plot_results
from tests.num_points_vs_deviation.write_to_file import write_to_file

def test_integrand0(x):
    return np.exp(-x)*x**2

def test_integrand1(x):
    return np.exp(-x**2/2)

def test_integrand2(x):
    return np.cos(x) * np.sin(x) * np.exp(-x)

def test_integrand3(x):
    return 1/(1+x**2) 

def deviation_all_degrees(integrand):
    # Compute integrals with Lebedev quadrature. The integration ranges are fixed to xi_range and phi_range 
    gauss_laguerre = GaussLaguerre()
    num_points_deviation_dict = {}
    TI = integrate.quad(integrand, 0, np.inf)[0]
    # calculate the deviation of the integral for all available degrees
    for num_gridpoints in range(1, 15):
        TI_laguerre = gauss_laguerre.integrate_function(integrand, num_gridpoints)
        # calculate the deviation in per cent
        deviation = abs(100*(TI-TI_laguerre)/TI)
        # add result to dictionary
        num_points_deviation_dict[num_gridpoints] = deviation
    return num_points_deviation_dict

num_points_deviation_dicts = []
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand0))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand1))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand2))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand3))
function_labels = ["f(x) = x²*exp(-x)", "f(x) = exp(-x²/2)", "f(x) = cos(x)sin(x)exp(-x)", "f(x) = 1/(1+x²)"]
write_to_file("tests/num_points_vs_deviation/results/gauss_laguerre.txt", num_points_deviation_dicts, function_labels)
description = "Test functions were integrated on the semi-finite interval [0, inf) using Gauss-Laguerre grids of different sizes."
description += "\n\nReference values: results of scipy quadrature."
plot_results(num_points_deviation_dicts, function_labels, description)