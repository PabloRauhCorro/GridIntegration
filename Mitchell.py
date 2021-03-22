
import numpy as np
from scipy.spatial.transform import Rotation
from GridIntegration import GridIntegration

class Mitchell_Integration(GridIntegration):

    def get_points(self, num_gridpoints):
        # read out written grid
        resolution, ngp2 = self.calculate_resolution(num_gridpoints)
        filename = "written_grids/mitchell/simple_grid_%s.qua" % str(resolution)
        quat = np.loadtxt(filename)
        RO1 = Rotation.from_quat(quat)
        gridpoints = RO1.as_euler('zxz', degrees=False)
        gridpoints = self.format_angles(gridpoints)
        return gridpoints
    
    def get_weights(self):
        return 8 * np.pi *np.pi
    
    def integrate_function(self, function, num_gridpoints):
        gridpoints = self.get_points(num_gridpoints)
        weight = self.get_weights()
        alpha, beta, gamma = gridpoints[:,0], gridpoints[:,1], gridpoints[:,2]
        return weight * np.sum(function(alpha, beta, gamma))


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
        for i in range(gridpoints.shape[0]):
            if gridpoints[i][0] < 0:
                gridpoints[i][0] = gridpoints[i][0] + 2*np.pi
            if gridpoints[i][1] < 0:
                gridpoints[i][1] = -gridpoints[i][1]
            if gridpoints[i][2] < 0:
                gridpoints[i][2] = gridpoints[i][2] + 2*np.pi
        return gridpoints
