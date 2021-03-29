import numpy as np
from math import ceil
from GridIntegration import GridIntegration
from written_grids.Lebedev_grid import lebedevdictionary
from supplementary.coordinate_system_conversions import cartesian2spherical

# used to calculate integrals of functions on the suface of the unit sphere. 
# The functions need to have this shape: g(r,θ,φ) = sin(θ) * f(r,θ,φ)
# ∫sinθ dθ ∫f(r,θ,φ) dφ = ∑ wi f(r, θi, φi) 
# the integration ranges are: theta = [0, pi], phi = [0, 2pi]


class LebedevAngularIntegration(GridIntegration):

    def __init__(self):
        self.lebedev_grid = lebedevdictionary()
        self.available_degrees = self.lebedev_grid.keys()

    def get_number_of_points(self, degree):
        while(degree not in self.available_degrees):
            degree -= 1
        return ceil(1/3 * (degree+1)**2)

    def get_points(self, degree):
        return self.lebedev_grid[degree][:, 0:3]

    def get_weights(self, degree):
        return self.lebedev_grid[degree][:, 3]

    def integrate_function(self, function, degree):
        while(degree not in self.available_degrees):
            degree -= 1
        xyz_points = self.get_points(degree)
        weights = self.get_weights(degree)
        r, theta, phi = cartesian2spherical(xyz_points)
        return np.sum(weights * function(theta, phi))
