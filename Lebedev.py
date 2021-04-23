import numpy as np
from math import ceil
from GridIntegration import GridIntegration
from written_grids.Lebedev_grid import lebedevdictionary, lebedev_numpoints_degree
from supplementary.coordinate_system_conversions import cartesian2spherical

# used to calculate integrals of functions on the suface of the unit sphere. 
# The functions need to have this shape: g(r,θ,φ) = sin(θ) * f(r,θ,φ)
# ∫sinθ dθ ∫f(r,θ,φ) dφ = ∑ wi f(r, θi, φi) 
# the integration ranges are: theta = [0, pi], phi = [0, 2pi]


class LebedevAngularIntegration(GridIntegration):

    def __init__(self):
        self.lebedev_grid = lebedevdictionary()
        self.numpoints_degree_dict = lebedev_numpoints_degree()

    def get_points(self, num_gridpoints):
        # points in cartesian coordinates
        degree = self.degree_from_num_gridpoints(num_gridpoints)
        return self.lebedev_grid[degree][:, 0:3]
    
    def get_points_spherical(self, num_gridpoints):
        xyz_points = self.get_points(num_gridpoints)
        r, theta, phi = cartesian2spherical(xyz_points)
        return np.vstack((theta, phi))

    def get_weights(self, num_gridpoints):
        degree = self.degree_from_num_gridpoints(num_gridpoints)
        return self.lebedev_grid[degree][:, 3]

    def find_nearest(self, target):
        return min(self.numpoints_degree_dict.keys(), key = lambda num_point: abs(num_point-target))

    def degree_from_num_gridpoints(self, num_gridpoints):
        num_gridpoints = self.find_nearest(num_gridpoints)
        degree = self.numpoints_degree_dict[num_gridpoints]
        return degree

    def get_weighted_summands(self, function, num_gridpoints):
        xyz_points = self.get_points(num_gridpoints)
        weights = self.get_weights(num_gridpoints)
        r, theta, phi = cartesian2spherical(xyz_points)
        return weights*function(theta, phi)

    def integrate_function(self, function, num_gridpoints):
        return np.sum(self.get_weighted_summands( function, num_gridpoints))

    