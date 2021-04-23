
from functools import partial
import sys
import os
from scipy import integrate
import numpy as np
sys.path.append(os.getcwd())
from Lebedev import LebedevAngularIntegration
from supplementary.deg2rad import deg2rad


# normalized uniform distribution, area is 1
def uniform_distribution(xi, phi, mean, width):
    return np.where((xi >= mean-0.5*width) & (xi <= mean+0.5*width), 1/width, 0.0)


lebedev = LebedevAngularIntegration()

uniform_distr = partial(uniform_distribution, mean = 60 *deg2rad, width =10*deg2rad)

print("The uniform distributions are normalized to 1")
print("num_points  -   result")
print("Puni, mean = 60°, width = 10°")
for num_points in range(1, 20):
    print(str(num_points)+" - "+str(lebedev.integrate_function(uniform_distr, num_points)))
print()

uniform_distr = partial(uniform_distribution, mean = 45 *deg2rad, width =20*deg2rad)
for num_points in range(1, 20):
    print(str(num_points)+" - "+str(lebedev.integrate_function(uniform_distr, num_points)))