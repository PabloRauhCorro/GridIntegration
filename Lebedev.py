import numpy as np
from math import ceil
from GridIntegration import GridIntegration
from written_grids.Lebedev_grid import lebedevdictionary
from supplementary.coordinate_system_conversions import cartesian2spherical


class LebedevAngularIntegration(GridIntegration):

    def __init__(self):
        self.lebedev_grid = lebedevdictionary()

    def get_number_of_points(self, degree):
        return ceil(1/3 * (degree+1)**2)

    def get_points(self, degree):
        return self.lebedev_grid[degree][:, 0:3]

    def get_weights(self, degree):
        return self.lebedev_grid[degree][:, 3]

    def integrate_function(self, function, degree):
        xyz_points = self.get_points(degree)
        weights = self.get_weights(degree)
        r, theta, phi = cartesian2spherical(xyz_points)
        return np.sum(weights * function(theta, phi))
