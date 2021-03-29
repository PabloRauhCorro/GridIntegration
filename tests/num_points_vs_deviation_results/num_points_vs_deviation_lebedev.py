import sys, os
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration
from supplementary.von_mises_distribution import von_mises_distr
from supplementary.deg2rad import deg2rad

def test_integrand4(xi, phi, weighted=False):
    xi_mean = 45 *deg2rad
    xi_std = 10 * deg2rad
    phi_mean = 45 *deg2rad
    phi_std = 10 * deg2rad
    function = von_mises_distr(xi, xi_mean, xi_std) * von_mises_distr(phi, phi_mean, phi_std)
    return function if not weighted else function * np.sin(xi)

def scipy_result(integrand):
    # Compute integrals with dblquad and weight factor
    xi_range = [0.0, np.pi] 
    phi_range = [0, 2*np.pi]
    TI4 = integrate.dblquad(integrand, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1],args=(True,))[0]
    return TI4

def deviation_all_degrees(integrand):
    # Compute integrals with Lebedev quadrature. The integration ranges are fixed to xi_range and phi_range 
    lebedevAngularIntegration = LebedevAngularIntegration()
    num_points_deviation_dict = {}
    TI4 = scipy_result(integrand)
    # calculate the deviation of the integral for all available degrees
    for degree in lebedevAngularIntegration.available_degrees:
        TI4_lebedev = lebedevAngularIntegration.integrate_function(integrand, degree)
        # calculate the deviation in per cent
        deviation = abs(100*(TI4-TI4_lebedev)/TI4)
        # add result to dictionary
        num_points_deviation_dict[lebedevAngularIntegration.get_number_of_points(degree)] = deviation
    return num_points_deviation_dict


def write_to_file(filename, num_points_deviation_dict):
    with open(filename, 'w') as file:
        file.write("Test integrand: f(ξ,φ) = P_vonmises(ξ) * P_vonmises(φ)\n\n")
        file.write("num points    deviation in %\n")
        for num_points, deviation in num_points_deviation_dict.items():
            file.write(str(num_points)+"    "+str(deviation)+"\n")

def plot_results(results_dict):
    fig = plt.figure()
    axes = fig.add_subplot()
    axes.set_ylabel('deviation')
    axes.set_xlabel('number of gridpoints')
    plt.plot(num_points_deviation_dict.keys(), num_points_deviation_dict.values(), marker = ".")
    plt.show()

num_points_deviation_dict = deviation_all_degrees(test_integrand4)
write_to_file("tests/num_points_vs_deviation_results/lebedev.txt", num_points_deviation_dict)
plot_results(num_points_deviation_dict)