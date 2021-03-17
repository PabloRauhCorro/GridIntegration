from scipy.integrate import nquad
from GridIntegration import GridIntegration

class Gauss_Legendre(GridIntegration):

    """ 
    Gauss-Legendre-Quadrature. The scipy implementation of this technique is used.
    """

    def get_weights(self, dimension, integration_ranges):
        pass
    def get_points(self, dimension, integration_ranges):
        pass
    def integrate_function(self, function, dimension, integration_ranges):
        return nquad(function, ranges=integration_ranges)[0]
