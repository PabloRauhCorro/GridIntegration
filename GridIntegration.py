import numpy as np
from abc import ABC, abstractmethod


class GridIntegration(ABC):
        
    @abstractmethod
    def get_weights(self, dimension, integration_ranges):
        pass

    @abstractmethod
    def get_points(self, dimension, integration_ranges):
        pass
    
    @abstractmethod
    def integrate_function(self, function, dimension, integration_ranges):
        pass

    
    """ def integrate_function(self, function, dimension, integration_ranges):
        if integration_ranges is not None and dimension != len(integration_ranges):
            raise ValueError("Integration ranges don't match number of dimensions.")
        weights = self.get_weights(dimension, integration_ranges)
        points = self.get_points(dimension, integration_ranges)
        return np.sum(function(points) * weights) """