
import numpy as np
from GridIntegration import GridIntegration

class Gauss_Legendre(GridIntegration):

    """ 
    Gauss-Legendre-Quadrature for a one dimensional finite integral with variable integration bounds
    """

    def get_weights(self, num_points, lower_bound, upper_bound):
        weights = np.polynomial.legendre.leggauss(num_points)[1]
        weights = weights * (upper_bound-lower_bound)/2
        return weights

    def get_points(self, num_points, lower_bound, upper_bound):
        points = np.polynomial.legendre.leggauss(num_points)[0]
        #variable transformation for change of interval [-1,1]
        points = (upper_bound-lower_bound)/2 * points + (upper_bound+lower_bound)/2
        return points

    def get_weighted_summands(self, function, num_points, lower_bound, upper_bound):
        weights = self.get_weights(num_points, lower_bound, upper_bound)
        points = self.get_points(num_points, lower_bound, upper_bound)
        return weights * function(points)

    def integrate_function(self, function, num_points, lower_bound, upper_bound):
        return np.sum(self.get_weighted_summands(function, num_points, lower_bound, upper_bound))
