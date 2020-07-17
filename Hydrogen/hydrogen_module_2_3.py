# Hydrogen
# Checker (2) and velocity extractor functions (3)
# all lengths in cm
# all times in s

import numpy as np


# np.loadtxt(fname)

STATE = np.array((200000,50,70,0,0,)) # Tester State; remove in final
STATE2 = np.array((100000,-30,40,0,0)) # Tester State; remove in final

def filter(particles, aperture_diameter = 0.5, axial_length = 30):
    '''
    Filters out particles that would not reach the aperture and returns the ones which DO.

    particles: array containing some 5x1 vectors representing particles (vz, vy, z, y, vx)
    aperture_diameter: diameter of terminal aperture, in cm
    axial_length: distance that particles must travel in x direction to reach aperture
    '''
    radius = aperture_diameter / 2 # convert diameter to radius

    vx, vy, vz, y0, z0  = particles.T # unpack each particle

    T = axial_length / vx # get Times taken to traverse the x distance
    z = z0 + vz * T # get final Z at time T
    y = y0 + vy * T # get final Y at time T

    Zmask = np.nonzero(np.abs(z) < radius) # indices where Z is okay
    Ymask = np.nonzero(np.abs(y) < radius) # indices where Y is okay

    return particles[np.intersect1d(Zmask, Ymask)] # takes indices where both Y and Z are okay

print(filter(np.array([STATE, STATE2])))
    
def positions(particles, axial_length = 30, num_steps = 500):
    '''
    Gets the position over time functions of particles.

    Timescale is based on the slowest particle in the set; some particles WILL overshoot

    particles: array containing some 5x1 vectors representing particles (vx, vy, vz, y0, z0)
    axial_length: distance that particles must travel in x direction to reach aperture
    num_steps: number of desired timesteps for position
    '''
    vx, vy, vz, y0, z0  = particles.T # unpack each particle

    T = axial_length / np.amin(vx) # get Time taken to traverse the x distance, on the SLOWEST particle

    times = np.tile(np.linspace(0, T, num=num_steps), (particles.shape[0], 1)).T # make range of times, and stack them up to work on the array

    x = vx * times
    y = y0 + vy * times
    z = z0 + vz * times
    # apply newton's law

    return np.transpose(np.array((x, y, z)), [2, 0, 1])

print(positions(np.array([STATE, STATE2]), num_steps=4))
