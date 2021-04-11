
import numpy as np
from scipy.spatial.transform import Rotation
from GridIntegration import GridIntegration

class Mitchell_Integration(GridIntegration):

    def get_points(self, num_gridpoints):
        resolution, ngp2 = self.calculate_resolution(num_gridpoints)
        filename = "written_grids/mitchell/simple_grid_%s.qua" % str(resolution)
        quat = np.loadtxt(filename)
        RO1 = Rotation.from_quat(quat)
        points = RO1.as_euler('zxz', degrees=False)
        points = self.format_angles(points)
        return points
    
    def get_weights(self, num_gridpoints):
        return (8 * np.pi *np.pi)/num_gridpoints

    def get_weighted_summands(self, function, num_gridpoints):
        gridpoints = self.get_points(num_gridpoints)
        weight = self.get_weights(gridpoints.shape[0])
        alpha, beta, gamma = gridpoints[:,0], gridpoints[:,1], gridpoints[:,2]
        return weight * function(alpha, beta, gamma)

    def integrate_function(self, function, num_gridpoints):
        return np.sum(self.get_weighted_summands(function, num_gridpoints))

    def calculate_resolution(self, num_gridpoints):
        resolutions = np.linspace(0,10,11)
        ngps = 12 * 6 * 2**(3*resolutions)
        idx = self.find_nearest(ngps, num_gridpoints)
        return [idx, ngps[idx]]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def format_angles(self, gridpoints):
        gridpoints[:,0] = np.where(gridpoints[:,0] < 0, gridpoints[:,0] + 2*np.pi, gridpoints[:,0])
        gridpoints[:,1] = np.where(gridpoints[:,1] < 0, gridpoints[:,1] * -1, gridpoints[:,1])
        gridpoints[:,2] = np.where(gridpoints[:,2] < 0, gridpoints[:,2] + 2*np.pi, gridpoints[:,2])
        return gridpoints
