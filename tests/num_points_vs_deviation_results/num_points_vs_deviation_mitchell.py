
import sys, os
import numpy as np
from scipy.integrate import tplquad
sys.path.append('..')
sys.path.append(os.getcwd())
from Mitchell import Mitchell_Integration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad

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

def scipy_result(integrand):
    # Integration with tplquad
    alpha_range = [0.0, 2*np.pi] 
    beta_range = [0.0, np.pi]
    gamma_range = [0.0, 2*np.pi]
    TI3 = tplquad(integrand, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
                lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,))[0]
    return TI3 

def deviation_all_degrees(integrand):
    # Calculate integral with scipy
    TI3 = scipy_result(integrand)
    # Calculate integral using Mitchell's simple grid
    mitchell_integration = Mitchell_Integration()
    num_points_all = [75, 576, 4608, 36864]
    num_points_deviation_dict = {}
    for num_points in num_points_all:
        TI3_mitchell = mitchell_integration.integrate_function(test_integrand3, num_points)
        deviation = abs(100*(TI3-TI3_mitchell)/TI3)
        num_points_deviation_dict[num_points] = deviation
    return num_points_deviation_dict

def write_to_file(num_points_deviation_dict):
    with open("tests/num_points_vs_deviation_results/mitchell.txt", 'w') as file:
        file.write("Test integrand: f(α,β,ɣ) = P_vonmises(α)*P_vonmises(β)*P_vonmises(ɣ)\n\n")
        file.write("num points    deviation in %\n")
        for num_points, deviation in num_points_deviation_dict.items():
            file.write(str(num_points)+"    "+str(deviation)+"\n")

num_points_deviation_dict = deviation_all_degrees(test_integrand3)
write_to_file(num_points_deviation_dict)