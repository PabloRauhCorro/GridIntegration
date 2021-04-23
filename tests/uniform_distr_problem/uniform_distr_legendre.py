
from functools import partial
import sys
import os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Gauss_Legendre import Gauss_Legendre


# normalized uniform distribution, area is 1
def uniform_distribution(x, mean, width):
    return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0)


legendre = Gauss_Legendre()

uniform_distr = partial(uniform_distribution, mean = 0, width =1)
print("The uniform distributions are normalized to 1")
print("num_points  -   result")
print("Puni, mean = 0, width = 1, integrated on interval [-1, 1]")
for num_points in range(1, 20):
    print(str(num_points)+" - "+str(legendre.integrate_function(uniform_distr, num_points, lower_bound = -1, upper_bound= 1)))

print()
uniform_distr = partial(uniform_distribution, mean = 2.84, width =0.08)
print("Puni, mean = 2.84, width = 0.08, integrated on interval [2, 3]")
for num_points in range(1, 20):
    print(str(num_points)+" - "+str(legendre.integrate_function(uniform_distr, num_points, lower_bound = 2, upper_bound= 3)))
print()

uniform_distr = partial(uniform_distribution, mean = 2.84, width =0.08)
print("Puni, mean = 2.84, width = 0.08, integrated on interval [2.76, 2.92]")
for num_points in range(1, 20):
    print(str(num_points)+" - "+str(legendre.integrate_function(uniform_distr, num_points, lower_bound = 2.76, upper_bound= 2.92)))
