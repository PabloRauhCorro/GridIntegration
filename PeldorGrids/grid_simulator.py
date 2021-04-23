import sys, os
import  numpy as np
from scipy.spatial.transform import Rotation
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration
from Gauss_Legendre import Gauss_Legendre
from Mitchell import Mitchell_Integration
from PeldorGrids.distributions import distributions, deg2rad


class GridSimulator():

    def __init__(self):
        self.separate_grids = True
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_orientations = []
        self.effective_gfactors_spin1 = []
        self.detection_probabilities_spin1 = {}
        self.pump_probabilities_spin1 = {}

        self.grid_sizes = {"L":38, "N":11, "M":230, "K":4608, "O":11}
        self.lebedev = LebedevAngularIntegration()
        self.gauss_leg = Gauss_Legendre()
        self.mitchell = Mitchell_Integration()

    def set_field_orientations_grid(self, treshold):
        ''' Powder-averging grid via Lebedev angular quadrature (L) '''
        # orientations of the field vector in cartesian coordinates (points on a unit sphere)
        field_orientations = self.lebedev.get_points(self.grid_sizes["L"])
        field_weights = self.lebedev.get_weighted_summands((lambda xi, phi: np.sin(xi)), self.grid_sizes["L"])
        field_orientations = field_orientations[field_weights>treshold] 
        field_weights = field_weights[field_weights>treshold]
        return field_orientations

    def set_r_values_grid(self, r_mean, r_width, r_max, r_min, treshold):
        ''' r-grid via Gauss-Legendre quadrature '''
        r_values = self.gauss_leg.get_points(self.grid_sizes["N"], r_min, r_max)
        function = lambda r: distributions['r'](r, r_mean, r_width)
        r_values_weights = self.gauss_leg.get_weighted_summands(function, self.grid_sizes["N"], r_min, r_max)
        r_values = r_values[r_values_weights>treshold]
        r_values_weights = r_values_weights[r_values_weights>treshold]
        return r_values

    def set_r_orientations_grid(self, xi_mean, xi_width, phi_mean, phi_width, treshold):
        ''' 
        xi/phi-grid via Lebedev angular quadrature. (M)
        It is  used to compute the orientations of the distance vector in the reference frame
        '''
        # unit vector of the r_orientation in cartesian coordinates
        r_orientations = self.lebedev.get_points(self.grid_sizes['M'])
        function = lambda xi, phi : (np.sin(xi)*distributions['xi'](xi, xi_mean, xi_width)*distributions['phi'](phi, phi_mean, phi_width))
        r_orientations_weights = self.lebedev.get_weighted_summands(function, self.grid_sizes['M'])
        r_orientations = r_orientations[r_orientations_weights>treshold]
        r_orientations_weights = r_orientations_weights[r_orientations_weights>treshold]
        return r_orientations

    def set_spin_frame_rotations_grid(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, treshold):
        '''
        alpha/beta/gamma-grid via Mitchell grid.
        It is used to compute rotation matrices transforming the reference frame into the spin frame
        '''
        alpha_beta_gamma = self.mitchell.get_points(self.grid_sizes['K'])
        function = (lambda alpha, beta, gamma: np.sin(beta)*distributions['alpha'](alpha, alpha_mean, alpha_width)*
        distributions['beta'](beta, beta_mean, beta_width)* distributions['gamma'](gamma, gamma_mean, gamma_width))
        alpha_beta_gamma_weights = self.mitchell.get_weighted_summands(function, self.grid_sizes['K'])
        alpha_beta_gamma = alpha_beta_gamma[alpha_beta_gamma_weights>treshold]
        alpha_beta_gamma_weights = alpha_beta_gamma_weights[alpha_beta_gamma_weights>treshold]
        # take convention from calculation settings later
        spin_frame_rotations = Rotation.from_euler('ZXZ', alpha_beta_gamma)
        return spin_frame_rotations
        
    def set_j_values_grid(self, j_mean, j_width, j_min, j_max, treshold):
        ''' j-grid via Gauss-Legendre quadrature '''
        j_values = self.gauss_leg.get_points(self.grid_sizes["N"], j_min, j_max)
        function = lambda r: distributions['r'](r, j_mean, j_width)
        j_values_weights = self.gauss_leg.get_weighted_summands(function, self.grid_sizes["N"], j_min, j_max)
        j_values = j_values[j_values_weights>treshold]
        j_values_weights = j_values_weights[j_values_weights>treshold]
        return j_values
        
    def set_coordinates(self, r_mean, r_width, r_min, r_max, xi_mean, xi_width, phi_mean, phi_width, treshold):
        ''' 
        The values of r, xi, and phi from corresponding distributions P(r), P(xi), and P(phi)
        are used to compute the coordinates of the distance vector in the reference frame
        '''
        # PROBLEM: different dimensions due to different grid sizes..
        r_values = self.set_r_values_grid(r_mean, r_width, r_max, r_min, treshold)
        r_orientation_cartesian = self.set_r_orientations_grid(xi_mean, xi_width, phi_mean, phi_width, treshold)
        # scale orientation_unit_vector with it's corresponding length 
        #coordinates = r_orientation_cartesian * r_values.reshape(r_values.size, 1)
        #return coordinates

    def compute_time_trace_via_grids(self, experiment, spins, variables, idx_spin1=0, idx_spin2=1, display_messages=False):
        ''' Computes a PDS time trace via integration grids '''
        # in progress
        if self.field_orientations == []:
            self.field_orientations
        r_values = self.set_r_values_grid(variables['r_mean'][0], variables['r_width'][0], variables['r_min'][0], variables['r_max'][0], treshold=0.001 )
        r_orientations = self.set_r_orientations_grid(variables['xi_mean'][0], variables['xi_width'][0], 
                                                 variables['phi_mean'][0], variables['phi_width'][0], 
                                                 treshold = 0.001)
        spin_frame_rotations_spin2 = self.set_spin_frame_rotations_grid(variables['alpha_mean'][0], variables['alpha_width'][0], 
                                                                   variables['beta_mean'][0], variables['beta_width'][0], 
                                                                   variables['gamma_mean'][0], variables['gamma_width'][0], 
                                                                   treshold = 1e-6)
        j_values = self.set_j_values_grid(variables['j_mean'][0], variables['j_width'][0], treshold=0)
        field_orientations_spin1 = self.field_orientations
        #field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin2, self.separate_grids)

# test 
grid_simulator = GridSimulator()
grid_simulator.set_field_orientations_grid(treshold = 0.3)
grid_simulator.set_r_values_grid(r_mean=1.94, r_width=0.03, r_min=1.94-4*0.03, r_max=1.94+4*0.03, treshold=0.001)
grid_simulator.set_r_orientations_grid(xi_mean=90*deg2rad, xi_width=10*deg2rad, phi_mean=180, phi_width=5, treshold = 0.001)
grid_simulator.set_spin_frame_rotations_grid(180*deg2rad, 20*deg2rad, 45*deg2rad, 20*deg2rad, 0, 20*deg2rad, treshold = 1e-6)
grid_simulator.set_coordinates(r_mean=1.94, r_width=0.03, r_min=1.94-4*0.03, r_max=1.94+4*0.03, xi_mean=90*deg2rad, xi_width=10*deg2rad, phi_mean=180, phi_width=5, treshold = 0.001)