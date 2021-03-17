import sys, os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration





# test functions
def test_integrand0(theta, phi):
    return np.sin(theta) * (phi+theta)
def test_integrand1(theta, phi):
    return np.sin(theta) * theta**2

# Compute integrals with Lebedev quadrature. The integration ranges are fixed to theta_range ad phi_range (I think)
lebedevAngularIntegration = LebedevAngularIntegration()
degree = 15
print("Lebedev: ")
print("Test integrand 0: " + str(lebedevAngularIntegration.integrate_function(test_integrand0, degree)))
print("Test integrand 1: " + str(lebedevAngularIntegration.integrate_function(test_integrand1, degree)))
print("\n")


# Compare to the results obtained with dblquad
theta_range = [0.0, np.pi] 
phi_range = [0.0, 2*np.pi]
print("Scipy dblquad:")
print("Test integrand 0: " + str(integrate.dblquad(test_integrand0, phi_range[0], phi_range[1], lambda phi: theta_range[0], lambda phi: theta_range[1])[0]))
print("Test integrand 0: " + str(integrate.dblquad(test_integrand1, phi_range[0], phi_range[1], lambda phi: theta_range[0], lambda phi: theta_range[1])[0]))