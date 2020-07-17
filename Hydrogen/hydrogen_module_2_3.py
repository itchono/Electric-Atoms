# Hydrogen
# Checker (2) and velocity extractor functions (3)
# all lengths in cm
# all times in s

import numpy as np


# np.loadtxt(fname)

STATE = np.array((50,70,0,0,200000)) # Tester State; remove in final
STATE2 = np.array((-30,40,0,0,100000)) # Tester State; remove in final

def filter(particles, aperture_diameter = 0.5, axial_length = 30):
    '''
    Filters out particles that would not reach the aperture.

    particles: array containing some 5x1 vectors representing particles (vz, vy, z, y, vx)
    aperture_diameter: diameter of terminal aperture, in cm
    axial_length: distance that particles must travel in x direction to reach aperture
    '''
    radius = aperture_diameter / 2 # convert diameter to radius

    vz, vy, z, y, vx = particles.T # unpack each particle

    T = axial_length / vx # get Times taken to traverse the x distance
    Z = z + vz * T # get final Z at time T
    Y = y + vy * T # get final Y at time T

    Zmask = np.nonzero(np.abs(Z) < radius) # indices where Z is okay
    Ymask = np.nonzero(np.abs(Y) < radius) # indices where Y is okay

    return particles[np.intersect1d(Zmask, Ymask)] # takes indices where both Y and Z are okay

print(filter(np.array([STATE]*5)))
    
def positions(particles, axial_length = 30, num_steps = 500):
    '''
    Gets the position over time functions of particles

    particles: array containing some 5x1 vectors representing particles (vz, vy, z, y, vx)
    axial_length: distance that particles must travel in x direction to reach aperture
    num_steps: number of desired timesteps for position
    '''

    vz, vy, z, y, vx = particles.T # unpack each particle

    T = axial_length / vx # get Times taken to traverse the x distance

    times = np.linspace(0, T, num=num_steps)

    X = vx * times
    Y = y + vy * times
    Z = z + vz * times
    # apply newton's law

    print("hello", X)

    #return np.array([X, Y, Z])

print(positions(np.array([STATE]*2), num_steps=10))
