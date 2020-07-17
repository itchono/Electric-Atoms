# Hydrogen
# Checker and velocity extractor functions
# all lengths in cm
# all times in s

import numpy as np

vz, vy, z, y, vx = 0,0,0,0,200000

# np.loadtxt(fname)

STATE = np.array((vz, vy, z, y, vx))


def filter(aperture_diameter = 0.5, axial_length = 30, particles = np.array([STATE]*5)):
    '''
    aperture_diameter: diameter of terminal aperture, in mm
    axial_length: distance that particles must travel in x direction to reach aperture
    particles: array of 5x1 vectors representing particles
    '''

    radius = aperture_diameter / 2

    '''
    vz, vy, z, y, vx = STATE

    T = axial_length / vx
    Z = z + vz * T
    Y = y + vy * T

    check = -radius < Z < radius and -radius < Y < radius
    '''

    T = axial_length / particles[:,4] # get Times taken to traverse the x distance
    Z = particles[:,2] + particles[:,0] * T # get final Z at time T
    Y = particles[:,3] + particles[:,1] * T # get final Y at time T

    return particles[np.nonzero(-radius < Z < radius and -radius < Y < radius)]

print(filter())
    

    
