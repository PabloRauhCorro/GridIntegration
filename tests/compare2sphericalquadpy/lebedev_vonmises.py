import numpy as np
import sys, os
sys.path.append(os.getcwd())
from scipy.special import i0
from Lebedev import LebedevAngularIntegration
from supplementary.distributions import von_mises_distr
from supplementary.deg2rad import deg2rad

def deviation_all_degrees(integrand):
    numpoints = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974]
    results = []
    lebedev = LebedevAngularIntegration()
    for points in numpoints:
        results.append(lebedev.integrate_function(integrand, points))
    return dict(zip(numpoints, results))

def test_integrand0(xi, phi):
    xi_mean = 45 *deg2rad
    xi_std = 10 * deg2rad
    phi_mean = 45 *deg2rad
    phi_std = 10 * deg2rad
    function = von_mises_distr(xi, xi_mean, xi_std) * von_mises_distr(phi, phi_mean, phi_std)
    return function 

results = deviation_all_degrees(test_integrand0)
with open('results_lebedev_vonmises.txt', 'w') as file:
    file.write("Integration of f(ξ,φ) = Pn(ξ) * Pn(φ), μ=45° σ=10° .\n\n")
    file.write("Number of gridpoints - result\n\n")
    for key, value in results.items():
        file.write(str(key) + " - " + str(value)+ "\n")