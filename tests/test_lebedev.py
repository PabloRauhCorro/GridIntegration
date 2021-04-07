import sys, os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad


# conversions:
# x = rsin(xi)cos(phi)
# y = rsin(xi)sin(phi)
# z = rcos(xi)

# test functions
# f(x,y,z) = x²
def test_integrand0(xi, phi, weighted = False):
    function = (np.sin(xi) * np.cos(phi))**2
    return function if not weighted else function * np.sin(xi)

# f(x,y,z) = exp(-z²)
def test_integrand1(xi, phi, weighted = False):
    function = np.exp(-np.cos(xi)**2)
    return function if not weighted else function * np.sin(xi)

# f(x,y,z) = x² + y² +z²
def test_integrand2(xi, phi, weighted = False):
    function = (np.sin(xi) * np.cos(phi))**2 + (np.sin(xi)*np.sin(phi))**2 + np.cos(xi)**2
    return function if not weighted else function * np.sin(xi)


# f(xi, phi) = Puni(xi)
def test_integrand3(xi, phi, weighted = False):
    function = (np.array(xi<0.7) * np.array(xi>0.3)).astype(int)
    return function if not weighted else int(xi<0.7 and xi>0.3) * np.sin(xi)

# f(xi, phi) = Pvon_mises(xi) * P_von_mises(phi)
def test_integrand4(xi, phi, weighted=False):
    xi_mean = 45 *deg2rad
    xi_std = 10 * deg2rad
    phi_mean = 45 *deg2rad
    phi_std = 10 * deg2rad
    function = von_mises_distr(xi, xi_mean, xi_std) * von_mises_distr(phi, phi_mean, phi_std)
    return function if not weighted else function * np.sin(xi)

# Compute integrals with dblquad and weight factor
xi_range = [0.0, np.pi] 
phi_range = [0, 2*np.pi]
TI0 = integrate.dblquad(test_integrand0, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]
TI1 = integrate.dblquad(test_integrand1, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]
TI2 = integrate.dblquad(test_integrand2, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]
TI3 = integrate.dblquad(test_integrand3, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]
TI4 = integrate.dblquad(test_integrand4, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]


#θ ξ φ
# Compute integrals with Lebedev quadrature. The integration ranges are fixed to xi_range and phi_range
lebedevAngularIntegration = LebedevAngularIntegration()
num_points = 972
TI0_lebedev = lebedevAngularIntegration.integrate_function(test_integrand0, num_points)
TI1_lebedev = lebedevAngularIntegration.integrate_function(test_integrand1, num_points)
TI2_lebedev = lebedevAngularIntegration.integrate_function(test_integrand2, num_points)
TI3_lebedev = lebedevAngularIntegration.integrate_function(test_integrand3, num_points)
TI4_lebedev = lebedevAngularIntegration.integrate_function(test_integrand4, num_points)


# Display results
print("Test functions without factor sin(ξ)")
print("Test integrand 0 : f(x,y,z) = x² ")
print("Test integrand 1 : f(x,y,z) = exp(-z²) ")
print("Test integrand 2 : f(x,y,z) = x² + y² +z² ")
print("Test integrand 3 : f(ξ,φ) = P_uni(xi))")
print("Test integrand 4 : f(ξ,φ) = P_vonmises(ξ) * P_vonmises(φ)")

print()

print("Scipy dblquad: ")
print("Test integrand 0 : " + str(TI0))
print("Test integrand 1 : " + str(TI1))
print("Test integrand 2 : " + str(TI2))
print("Test integrand 3 : " + str(TI3))
print("Test integrand 4 : " + str(TI4))

print()

print(f"Lebedev with {num_points} points: ")
print("Test Integrand 0: " + str(TI0_lebedev) + " - Deviation: " + str(abs(100*(TI0-TI0_lebedev)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_lebedev) + " - Deviation: " + str(abs(100*(TI1-TI1_lebedev)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_lebedev) + " - Deviation: " + str(abs(100*(TI2-TI2_lebedev)/TI2)) + "%")
print("Test Integrand 3: " + str(TI3_lebedev) + " - Deviation: " + str(abs(100*(TI3-TI3_lebedev)/TI3)) + "%")
print("Test Integrand 4: " + str(TI4_lebedev) + " - Deviation: " + str(abs(100*(TI4-TI4_lebedev)/TI4)) + "%")


