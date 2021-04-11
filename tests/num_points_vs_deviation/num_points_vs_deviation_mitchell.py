
import sys, os
import numpy as np
from scipy.integrate import tplquad
sys.path.append('..')
sys.path.append(os.getcwd())
from Mitchell import Mitchell_Integration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad
from tests.num_points_vs_deviation.plot_deviation import plot_results
from tests.num_points_vs_deviation.write_to_file import write_to_file

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

def scipy_result(integrand):
    # Integration with tplquad
    alpha_range = [0.0, 2*np.pi] 
    beta_range = [0.0, np.pi]
    gamma_range = [0.0, 2*np.pi]
    TI = tplquad(integrand, gamma_range[0], gamma_range[1], lambda gamma: beta_range[0], lambda gamma: beta_range[1], 
                lambda gamma, beta: alpha_range[0], lambda gamma, beta: alpha_range[1], args=(True,))[0]
    return TI

def deviation_all_degrees(integrand):
    # Calculate integral with scipy
    TI = scipy_result(integrand)
    # Calculate integral using Mitchell's simple grid
    mitchell_integration = Mitchell_Integration()
    num_points_all = [75, 576, 4608, 36864]
    num_points_deviation_dict = {}
    for num_points in num_points_all:
        TI_mitchell = mitchell_integration.integrate_function(integrand, num_points)
        deviation = abs(100*(TI-TI_mitchell)/TI)
        num_points_deviation_dict[num_points] = deviation
    return num_points_deviation_dict


num_points_deviation_dicts = []
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand0))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand1))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand2))
num_points_deviation_dicts.append(deviation_all_degrees(test_integrand3))
function_labels = ["f(α,β,ɣ) = cos²(β)*sin(β)", "f(α,β,ɣ) = (sin(α)+sin(β)+sin(ɣ))*sin(β)", "f(α,β,ɣ) = exp(α)sin(α)sin(β)", "f(α,β,ɣ) = Pn(α)Pn(β)Pn(ɣ)sin(β), μ=30° σ=10°"]
write_to_file("tests/num_points_vs_deviation/results/mitchell.txt", num_points_deviation_dicts, function_labels)
description = "Test functions were integrated on the intervals α=[0, 2π], β=[0, π], ɣ=[0, 2π]  using Mitchell grids of different sizes."
description += "\n\nReference values: results of scipy quadrature."
plot_results(num_points_deviation_dicts, function_labels, description)