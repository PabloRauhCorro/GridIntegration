import sys, os
import numpy as np
from scipy.integrate import tplquad
sys.path.append(os.getcwd())
from Mitchell import Mitchell_Integration

# test integrands

def test_integrand1(alpha, beta, gamma):
    return np.sin(beta) * np.cos(beta)**2
def test_integrand2(alpha, beta, gamma):
    return np.sin(alpha) * np.sin(beta) * np.sin(gamma)

# Integration with tplquad
alpha_range = [0.0, 2*np.pi] 
beta_range = [0.0, np.pi]
gamma_range = [0.0, 2*np.pi]
 
TI1 = tplquad(test_integrand1, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
            lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1] )[0]             
TI2 = tplquad(test_integrand2, 
				  gamma_range[0], gamma_range[1], 
				  lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
				  lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1])[0]

print("Scipy TplQuad: ")
print("Test Integrand 1: " +  str(TI1))
print("Test Integrand 2: " +  str(TI2))
print("\n")

# Calculate integral using Mitchell's simple grid
mitchell_integration = Mitchell_Integration()

TI1_mitchell = mitchell_integration.integrate_function(test_integrand1, 1000)
TI2_mitchell = mitchell_integration.integrate_function(test_integrand2, 1000)
print("Integration using Mitchell simple grid: ")

print("Test Integrand 1: " + str(TI1_mitchell) + " - Deviation: " + str(100*(TI1-TI1_mitchell)/TI1) + "%")
print("Test Integrand 2: " + str(TI2_mitchell) + " - Deviation: " + str(100*(TI2-TI2_mitchell)/TI2) + "%")
print("\n")


