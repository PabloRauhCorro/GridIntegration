import sys, os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration





# test functions
# f(x,y,z) = x²
def test_integrand1(theta, phi, weighted = False):
    function = (np.sin(theta) * np.cos(phi))**2
    return function if not weighted else function * np.sin(theta)

# f(x,y,z) = exp(-z²)
def test_integrand2(theta, phi, weighted = False):
    function = np.exp(-np.cos(theta)**2)
    return function if not weighted else function * np.sin(theta)

# Compute integrals with Lebedev quadrature. The integration ranges are fixed to theta_range ad phi_range (I think)
lebedevAngularIntegration = LebedevAngularIntegration()
degree = 41
print("Lebedev: ")
print("Test integrand 2: " + str(lebedevAngularIntegration.integrate_function(test_integrand2, degree)))
print("Test integrand 1: " + str(lebedevAngularIntegration.integrate_function(test_integrand1, degree)))
print("\n")


# Compare to the results obtained with dblquad
theta_range = [0.0, np.pi] 
phi_range = [0, 2*np.pi]
print("Scipy dblquad:")
print("Test integrand 0: " + str(integrate.dblquad(test_integrand2, phi_range[0], phi_range[1], lambda phi: theta_range[0], lambda phi: theta_range[1], args=(True,))[0]))
print("Test integrand 1: " + str(integrate.dblquad(test_integrand1, phi_range[0], phi_range[1], lambda phi: theta_range[0], lambda phi: theta_range[1],args=(True,))[0]))