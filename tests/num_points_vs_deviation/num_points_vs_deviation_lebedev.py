import sys, os
import numpy as np
from scipy import integrate
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad
from tests.num_points_vs_deviation.plot_deviation import plot_results
from tests.num_points_vs_deviation.write_to_file import write_to_file

# f(ξ,φ) = Pn(ξ) * Pn(φ)"
def test_integrand0(xi, phi, weighted=False):
    xi_mean = 45 *deg2rad
    xi_std = 10 * deg2rad
    phi_mean = 45 *deg2rad
    phi_std = 10 * deg2rad
    function = von_mises_distr(xi, xi_mean, xi_std) * von_mises_distr(phi, phi_mean, phi_std)
    return function if not weighted else function * np.sin(xi)

# f(x,y,z) = exp(-z²)
def test_integrand1(xi, phi, weighted = False):
    function = np.exp(np.cos(xi)**2)
    return function if not weighted else function * np.sin(xi)

# f(x,y,z) = x² + y² +z²
def test_integrand2(xi, phi, weighted = False):
    function = (np.sin(xi) * np.cos(phi))**2 + (np.sin(xi)*np.sin(phi))**2 + np.cos(xi)**2
    return function if not weighted else function * np.sin(xi)

# f(xi, phi) = P_uni(xi)
def test_integrand3(xi, phi, weighted = False):
    function = (np.array(xi>0)* np.array(xi<np.pi/4)).astype(int)
    return function if not weighted else int(xi>0 and xi<np.pi/4) * np.sin(xi)


def scipy_result(integrand):
    # Compute integrals with dblquad and weight factor
    xi_range = [0.0, np.pi] 
    phi_range = [0, 2*np.pi]
    TI = integrate.dblquad(integrand, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]
    return TI

def deviation_all_degrees(integrand):
    # Compute integrals with Lebedev quadrature. The integration ranges are fixed to xi_range and phi_range 
    lebedevAngularIntegration = LebedevAngularIntegration()
    num_points_deviation_dict = {}
    TI = scipy_result(integrand)
    # calculate the deviation of the integral for all available degrees
    for degree in lebedevAngularIntegration.available_degrees:
        num_gridpoints = lebedevAngularIntegration.get_number_of_points(degree)
        if num_gridpoints > 1000:
            break
        TI_lebedev = lebedevAngularIntegration.integrate_function(integrand, degree)
        # calculate the deviation in per cent
        deviation = abs(100*(TI-TI_lebedev)/TI)
        # add result to dictionary
        num_points_deviation_dict[num_gridpoints] = deviation
    return num_points_deviation_dict



num_points_deviation_dicts = []
ti0 = deviation_all_degrees(test_integrand0)
ti0.pop(12)
num_points_deviation_dicts.append(ti0)
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand1))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand2))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand3))
function_labels = ["f(ξ,φ) = Pn(ξ) * Pn(φ), μ=45° σ=10°", "f(x,y,z) = exp(-z²)", "f(x,y,z) = x² + y² +z²", "f(ξ,φ) = 1 if 0<ξ<π/4 else 0"]
write_to_file("tests/num_points_vs_deviation/results/lebedev.txt", num_points_deviation_dicts, function_labels)
description = "Test functions were integrated on the unit sphere using Lebedev grids of different sizes.\nReference values: results of scipy quadrature."
plot_results(num_points_deviation_dicts, function_labels, description)