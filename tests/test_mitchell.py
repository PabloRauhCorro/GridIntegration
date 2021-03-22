import sys, os
import numpy as np
from scipy.integrate import tplquad
sys.path.append(os.getcwd())
from Mitchell import Mitchell_Integration

# test function
def test_integrand0(alpha, beta, gamma):
    return np.sin(alpha) * np.sin(beta) * np.sin(gamma)

def test_integrand1(alpha, beta, gamma):
    return np.sin(beta) * np.cos(beta)**2


# Calculate integral using Mitchell's simple grid
mitchell_integration = Mitchell_Integration()
TI0_mitchell = mitchell_integration.integrate_function(test_integrand0, 1000)
TI1_mitchell = mitchell_integration.integrate_function(test_integrand1, 1000)
print("Mitchell simple grid: ")
print("Test Integrand 0: " +   str(TI0_mitchell))
print("Test Integrand 1: " + str(TI1_mitchell))
print("\n")

#Compare to the results obtained with tplquad
alpha_range = [0.0, 2*np.pi] 
beta_range = [0.0, np.pi]
gamma_range = [0.0, 2*np.pi]
TI0 = tplquad(test_integrand0, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
            lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1] )[0] 
TI1 = tplquad(test_integrand1, 
				  gamma_range[0], gamma_range[1], 
				  lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
				  lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1])[0]
print("Scipy TplQuad: ")
print("Test Integrand 0: " +  str(TI0))
print("Test Integrand 1: " +  str(TI1))
print("\n")

# calculate the deviations
print("Deviation of Test Integrand 0:" + str(100*(TI0-TI0_mitchell)/TI0) + "%")
print("Deviation of Test Integrand 1:" + str(100*(TI1-TI1_mitchell)/TI1) + "%")