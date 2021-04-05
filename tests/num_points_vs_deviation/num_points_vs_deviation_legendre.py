import sys, os
import numpy as np
from scipy import integrate
sys.path.append(os.getcwd())
from Gauss_Legendre import Gauss_Legendre
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad
from tests.num_points_vs_deviation.plot_deviation import plot_results
from tests.num_points_vs_deviation.write_to_file import write_to_file

def test_integrand0(x):
    return (x**5 + x ** 3)/7*x**2

def test_integrand1(x):
    return np.sin(x)

def test_integrand2(x):
    return np.exp(-1/2 * x **2)

def test_integrand3(x, called_from_scipy = False):
    function_for_array = (np.array(x>0.2)* np.array(x<0.6)).astype(int)
    return function_for_array if not called_from_scipy else int(x>0.2 and x<0.6)



def deviation_all_degrees(integrand):
    # Compute integrals with Lebedev quadrature. The integration ranges are fixed to xi_range and phi_range 
    gauss_legendre = Gauss_Legendre()
    num_points_deviation_dict = {}
    TI = integrate.quad(integrand, 0, 1)[0]
    # calculate the deviation of the integral for all available degrees
    for num_gridpoints in range(1, 15):
        TI_legendre = gauss_legendre.integrate_function(integrand, num_gridpoints, 0, 1)
        # calculate the deviation in per cent
        deviation = abs(100*(TI-TI_legendre)/TI)
        # add result to dictionary
        num_points_deviation_dict[num_gridpoints] = deviation
    return num_points_deviation_dict

num_points_deviation_dicts = []
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand0))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand1))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand2))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand3))
function_labels = ["f(x) = (x⁵+x³)/(7x²)", "f(x) = sin(x)", "f(x) = exp(-x²/2)", "f(x) = 1 if 0.2<x<0.6 else 0"]
write_to_file("tests/num_points_vs_deviation/results/gauss_legendre.txt", num_points_deviation_dicts, function_labels)
description = "Test functions were integrated on the interval [0, 1] using Gauss-Legendre grids of different sizes."
description += "\n\nReference values: results of scipy quadrature."
plot_results(num_points_deviation_dicts, function_labels, description)