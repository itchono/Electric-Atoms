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

    Should be working well
    '''
    radius = aperture_diameter / 2 # convert diameter to radius

    vz, vy, z, y, vx = particles.T # unpack each particle

    T = axial_length / vx # get Times taken to traverse the x distance
    Z = z + vz * T # get final Z at time T
    Y = y + vy * T # get final Y at time T

    Zmask = np.nonzero(np.abs(Z) < radius) # indices where Z is okay
    Ymask = np.nonzero(np.abs(Y) < radius) # indices where Y is okay

    return particles[np.intersect1d(Zmask, Ymask)] # takes indices where both Y and Z are okay
    
def positions_fixed_stepnumber(particles, axial_length = 30, num_steps = 500):
    '''
    Gets the position over time functions of particles

    particles: array containing some 5x1 vectors representing particles (vz, vy, z, y, vx)
    axial_length: distance that particles must travel in x direction to reach aperture
    num_steps: number of desired timesteps for position

    Currently: Does not allow variable time steps
    '''
    
    vz, vy, z, y, vx = particles.T # unpack each particle

    T = axial_length / vx # get Times taken to traverse the x distance

    times = np.linspace(0, T, num=num_steps)

    X = vx * times
    Y = y + vy * times
    Z = z + vz * times
    # apply newton's law

    return np.transpose(np.array((X, Y, Z)), [2, 0, 1])

print(positions_fixed_stepnumber(np.array([STATE, STATE2]), num_steps=4))


def positions_fixed_timestep(particles, axial_length = 30, time_step = 1e-5):
    '''
    Gets the position over time functions of particles

    particles: array containing some 5x1 vectors representing particles (vz, vy, z, y, vx)
    axial_length: distance that particles must travel in x direction to reach aperture
    time_step: time step in microseconds

    Returns a slower python list :(
    '''

    results = []

    for p in particles:
        vz, vy, z, y, vx = p # unpack each particle

        T = axial_length / vx

        times = np.arange(0, T + time_step, time_step)

        X = vx * times
        Y = y + vy * times
        Z = z + vz * times
        # apply newton's law

        results.append(np.vstack((X, Y, Z)))

    return results

print(positions_fixed_timestep(np.array([STATE, STATE2])))