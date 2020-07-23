import numpy as np 
import pickle
import time
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import cProfile


def particleGenerator(NUM_POINTS):
    mu_vel_yz, sigma_vel_yz = 0, 200 # in m/s
    mu_vel_x, sigma_vel_x = 2000, 1000 # m/s
    minimum, maximum = -1e-3, 1e-3
    #Velocity Generation:

    v_z = np.random.normal(mu_vel_yz,sigma_vel_yz,NUM_POINTS)
    v_y = np.random.normal(mu_vel_yz,sigma_vel_yz,NUM_POINTS)
    v_x = np.random.normal(mu_vel_x,sigma_vel_x,NUM_POINTS)

    
    y_0 =  np.random.uniform(minimum,maximum,NUM_POINTS)
    z_0 =  np.random.uniform(minimum,maximum,NUM_POINTS)

    particles = np.array((v_x,v_y,v_z,y_0,z_0)).T

    np.savetxt('particles.txt',particles)

def getParticles():
    return np.loadtxt('particles.txt')

def filter(particles, aperture_diameter = 0.005, axial_length = 0.30):
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


    mask = np.nonzero(np.sqrt(z**2 + y**2) < radius) # indices where z and y are inside the aperture radius

    print(f"{len(mask[0])} of {len(particles)} made it through ({len(mask[0])/len(particles) * 100}%)")
    
    return particles[mask] # takes indices where both Y and Z are okay


def positionOverTime(particles, axial_length = 0.30, num_steps = 500):
    '''
    Gets the position over time functions of particles.

    Timescale is based on the slowest particle in the set; some particles WILL overshoot

    particles: array containing some 5x1 vectors representing particles (vx, vy, vz, y0, z0)
    axial_length: distance that particles must travel in x direction to reach aperture
    num_steps: number of desired timesteps for position
    '''
    vx, vy, vz, y0, z0  = particles.T # unpack each particle

    T = axial_length / np.amin(vx[np.nonzero(vx)]) # get Time taken to traverse the x distance, on the SLOWEST particle

    times = np.tile(np.linspace(0, T, num=num_steps), (particles.shape[0], 1)).T # make range of times, and stack them up to work on the array

    x = vx * times # in m 
    y = y0 + vy * times # in m
    z = z0 + vz * times # in m
    # apply newton's law

    return np.transpose(np.array((x, y, z)), [2, 0, 1])


def profiler():
    NUM_POINTS= int(20000)
    particleGenerator(NUM_POINTS)

    particles = getParticles()
    #print(particles)

    vx, vy, vz, y0, z0  = particles.T # unpack each particle

    T = 0.3 / vx # get Times taken to traverse the x distance
    z = z0 + vz * T # get final Z at time T
    y = y0 + vy * T # get final Y at time T


    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.scatter(z, vz)
    ax.set_xlabel("$z_{final}$ (mm)")
    ax.set_ylabel("$v_z$ (m/s)")

    ax = fig.add_subplot(222)
    ax.scatter(y, vy)
    ax.set_xlabel("$y_{final} (mm)$")
    ax.set_ylabel("$v_y$ (m/s)")


    filteredParticles = filter(particles)

    vx, vy, vz, y0, z0  = filteredParticles.T # unpack each particle

    T = 0.3 / vx # get Times taken to traverse the x distance
    z = z0 + vz * T # get final Z at time T
    y = y0 + vy * T # get final Y at time T

    ax = fig.add_subplot(223)
    ax.scatter(z, vz)
    ax.set_xlabel("$z_{final}$ (mm)")
    ax.set_ylabel("$v_z$ (m/s)")

    ax = fig.add_subplot(224)
    ax.scatter(y, vy)
    ax.set_xlabel("$y_{final}$ (mm)")
    ax.set_ylabel("$v_y$ (m/s)")

    fig.tight_layout(pad=1.0)


    trajectories = positionOverTime(filteredParticles)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')

    circley = 2.5 * np.cos(np.linspace(0, 2*np.pi, 30))
    circlez = 2.5 * np.sin(np.linspace(0, 2*np.pi, 30))
    circlex = np.array([0.3]*30)

    ax.plot(circlex, circley, circlez)
    
    ax.set_xlim3d([0, 0.4])
    ax.set_ylim3d([-4, 4])
    ax.set_zlim3d([-4, 4])

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    for i in trajectories:
        ax.plot(i[0,:], i[1,:]*1000, i[2,:]*1000, color="blue")


    plt.show()
    
if __name__ == "__main__":
    cProfile.run("profiler()")


