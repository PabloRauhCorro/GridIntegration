''' Coordinate system conversions '''
import numpy as np


def spherical2cartesian(rho, xi, phi):
    ''' Convert spherical coordinates into Cartestian coordinates '''
    x = rho * np.sin(xi) * np.cos(phi)
    y = rho * np.sin(xi) * np.sin(phi)
    z = rho * np.cos(xi)
    v = np.column_stack((x, y, z))
    return v


def cartesian2spherical(v):
    ''' Convert Cartesian coordinates into spherical coordinates '''
    xy = v[:,0]**2 + v[:,1]**2
    rho = np.sqrt(xy + v[:,2]**2)
    xi = np.arctan2(np.sqrt(xy), v[:,2])
    #xi = np.where(xi < 0, -xi, xi)
    phi = np.arctan2(v[:,1], v[:,0])
    #phi = np.where(phi < 0, phi + 2*np.pi, phi)
    return rho, xi, phi

""" def cartesian2spherical_(v):
    x = v[0]
    y = v[1]
    z = v[2]
    r = np.sqrt((x**2 + y**2 + z**2))
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y,x)
    return np.array([r, theta, phi]) """

