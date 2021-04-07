import numpy as np
from abc import ABC, abstractmethod


class GridIntegration(ABC):
        
    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_points(self):
        pass
    
    @abstractmethod
    def get_grid(self):
        pass
    
    @abstractmethod
    def integrate_function(self, function):
        pass

 