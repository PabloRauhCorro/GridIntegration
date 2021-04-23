import sys
import os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
try:
	import sphericalquadpy.lebedev
except ModuleNotFoundError:
	print("This file needs to be copied into the cloned sphericalquadpy module.")
	exit()
from scipy.special import i0



deg2rad = np.pi / 180.0


Q = sphericalquadpy.lebedev.Lebedev(nq = 200)

def f(x,y,z):
    return np.exp(-z**2)


def von_mises_distr(x, mean, width):
    kappa =  1 / width**2
    if np.isfinite(i0(kappa)):
        return np.exp(kappa * np.cos(x - mean)) / (2*np.pi * i0(kappa))
    else:
        return np.where(x == mean, 1.0, 0.0)

def test_integrand0(x, y, z):
    xi = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y,x)
    xi_mean = 45 *deg2rad
    xi_std = 10 * deg2rad
    phi_mean = 45 *deg2rad
    phi_std = 10 * deg2rad
    function = von_mises_distr(xi, xi_mean, xi_std) * von_mises_distr(phi, phi_mean, phi_std)
    return function 

print(Q.integrate(test_integrand0) )

def deviation_all_degrees(integrand):
    numpoints = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974]
    results = []
    for points in numpoints:
        Q = sphericalquadpy.lebedev.Lebedev(nq = points)
        results.append(Q.integrate(test_integrand0))
    return dict(zip(numpoints, results))


results = deviation_all_degrees(test_integrand0)

with open('results.txt', 'w') as file:
    file.write("Integration of f(ξ,φ) = Pn(ξ) * Pn(φ), μ=45° σ=10° using the sphericalquadpy module.\n\n")
    file.write("Number of gridpoints - result\n\n")
    for key, value in results.items():
        file.write(str(key) + " - " + str(value)+ "\n")




