
import sys, os
import numpy as np
from scipy.integrate import tplquad
sys.path.append('..')
sys.path.append(os.getcwd())
from Mitchell import Mitchell_Integration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad

# test integrands

def test_integrand0(alpha, beta, gamma, weighted = False):
    function = np.cos(beta)**2
    return function if not weighted else function * np.sin(beta)
    
def test_integrand1(alpha, beta, gamma, weighted = False):
    function = np.sin(alpha) + np.sin(gamma) + np.sin(beta)
    return function if not weighted else function * np.sin(beta)

def test_integrand2(alpha, beta, gamma, weighted = False):
    function = np.exp(-alpha)*np.sin(alpha) * np.sin(beta)
    return function if not weighted else function * np.sin(beta)

def test_integrand3(alpha, beta, gamma, weighted = False):
    alpha_mean = 30 * deg2rad
    beta_mean = 30 * deg2rad
    gamma_mean = 30 * deg2rad
    alpha_std = 10 * deg2rad
    beta_std = 10 * deg2rad
    gamma_std = 10 * deg2rad
    function = (von_mises_distr(alpha, alpha_mean, alpha_std) *
                von_mises_distr(beta, beta_mean, beta_std)*
                von_mises_distr(gamma, gamma_mean, gamma_std))
    return function if not weighted else function * np.sin(beta)



# Integration with tplquad
alpha_range = [0.0, 2*np.pi] 
beta_range = [0.0, np.pi]
gamma_range = [0.0, 2*np.pi]
TI0 = tplquad(test_integrand0, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
            lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,) )[0]          
TI1 = tplquad(test_integrand1, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
			lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,))[0] 
TI2 = tplquad(test_integrand2, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
			lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,))[0]
TI3 = tplquad(test_integrand3, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
			lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,))[0]

# Calculate integral using Mitchell's simple grid
mitchell_integration = Mitchell_Integration()
num_points = 4608
TI0_mitchell = mitchell_integration.integrate_function(test_integrand0, num_points)
TI1_mitchell = mitchell_integration.integrate_function(test_integrand1, num_points)
TI2_mitchell = mitchell_integration.integrate_function(test_integrand2, num_points)
TI3_mitchell = mitchell_integration.integrate_function(test_integrand3, num_points)

# Display results 
print("Test functions without factor sin(β)")
print("Test Integrand 0: f(α,β,ɣ) = cos(β)²")
print("Test Integrand 1: f(α,β,ɣ) = sin(α)+sin(β)+sin(ɣ)")
print("Test Integrand 2: f(α,β,ɣ) = exp(α)sin(α)sin(β)")
print("Test Integrand 3: f(α,β,ɣ) = P_vonmises(α)*P_vonmises(β)*P_vonmises(ɣ)")

print()

print("Scipy TplQuad: ")
print("Test Integrand 0: " +  str(TI0))
print("Test Integrand 1: " +  str(TI1))
print("Test Integrand 2: " +  str(TI2))
print("Test Integrand 3: " +  str(TI3))

print()

print(f"Integration using Mitchell simple grid with {num_points} gridpoints: ")
print("Test Integrand 0: " + str(TI0_mitchell) + " - Deviation: " + str(abs(100*(TI0-TI0_mitchell)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_mitchell) + " - Deviation: " + str(abs(100*(TI1-TI1_mitchell)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_mitchell) + " - Deviation: " + str(abs(100*(TI2-TI2_mitchell)/TI2)) + "%")
print("Test Integrand 3: " + str(TI3_mitchell) + " - Deviation: " + str(abs(100*(TI3-TI3_mitchell)/TI3)) + "%")
print()


