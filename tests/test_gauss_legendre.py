import sys
import os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Gauss_Legendre import Gauss_Legendre

# test functions
def test_integrand0(x):
    return (x**5 + x ** 3)/7*x**2

def test_integrand1(x):
    return np.sin(x)

def test_integrand2(x):
    return np.exp(-1/2 * x **2)

def test_integrand3(x, called_from_scipy = False):
    function_for_array = (np.array(x>0.2)* np.array(x<0.6)).astype(int)
    return function_for_array if not called_from_scipy else int(x>0.2 and x<0.6)

# compute integrals with scipy 
TI0 = integrate.quad(test_integrand0, 0, 1)[0]
TI1 = integrate.quad(test_integrand1, 0, 1)[0]
TI2 = integrate.quad(test_integrand2, 0, 1)[0]
TI3 = integrate.quad(test_integrand3, 0, 1)[0]



# compute integrals with gauss legendre quadrature
gauss_leg = Gauss_Legendre()
number_of_points =10
TI0_gauss_leg = gauss_leg.integrate_function(test_integrand0, number_of_points, 0, 1)
TI1_gauss_leg = gauss_leg.integrate_function(test_integrand1, number_of_points, 0, 1)
TI2_gauss_leg = gauss_leg.integrate_function(test_integrand2, number_of_points, 0, 1)
TI3_gauss_leg = gauss_leg.integrate_function(test_integrand3, number_of_points, 0, 1)

print("Test functions:")
print("Test Integrand 0: f(x) = (x⁵ + x³)/(7x²) , Integration from 0 to 1")
print("Test Integrand 1: f(x) = sin(x) , Integration from 0 to 1")
print("Test Integrand 2: f(x) = exp(-x²/2) , Integration from 0 to 1")
print("Test Integrand 3: f(x) = 1 if 0.2<x<0.6 else 0 , Integration from 0 to 1")

print("\nScipy quad:")
print("Test Integrand 0: " + str(TI0))
print("Test Integrand 1: " + str(TI1))
print("Test Integrand 2: " + str(TI2))
print("Test Integrand 2: " + str(TI3))

print(f"\nGauss Legendre with {number_of_points} points: ")
print("Test Integrand 0: " + str(TI0_gauss_leg)+ " - Deviation: " + str(abs(100*(TI0-TI0_gauss_leg)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_gauss_leg)+ " - Deviation: " + str(abs(100*(TI1-TI1_gauss_leg)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_gauss_leg)+ " - Deviation: " + str(abs(100*(TI2-TI2_gauss_leg)/TI2)) + "%")
print("Test Integrand 3: " + str(TI3_gauss_leg)+ " - Deviation: " + str(abs(100*(TI3-TI3_gauss_leg)/TI3)) + "%")
