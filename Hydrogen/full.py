import numpy as np 
import pickle
import time
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt


def particleGenerator(NUM_POINTS):
    mu_vel_yz, sigma_vel_yz = 0, 200 # in m/s
    mu_vel_x, sigma_vel_x = 2000, 1000 # m/s
    minimum, maximum = -1e-4, 1e-4
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

    print("z", z[np.abs(z) < 0.005])
    print("y", y[np.abs(y) < 0.005])

    mask = np.nonzero(np.sqrt(z**2 + y**2) < radius**2) # indices where z and y are inside the aperture radius
    
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

def oracleFunctionGenerator():

    startTime = time.perf_counter()
    x, y, z, HxRe, HxIm, HyRe, HyIm, HzRe, HzIm = np.loadtxt('HField.txt', skiprows = 2, unpack = True)
    endTime = time.perf_counter()

    print("Loading Time: ", str(endTime-startTime))

    startTime = time.perf_counter()



    H_x = NearestNDInterpolator((x,y,z),HxRe,"QJ")
    pickle.dump(H_x,open("H_x.function","wb"))


    H_y = NearestNDInterpolator((x,y,z),HyRe)
    pickle.dump(H_y,open("H_y.function","wb"))

    H_z = NearestNDInterpolator((x,y,z),HzRe)
    pickle.dump(H_z,open("H_z.function","wb"))

    endTime = time.perf_counter()


    print("Function Loading Time: ", str(endTime-startTime))

def getOracleFunctions():

    H_x = pickle.load(open("H_x.function","rb"))
    H_y = pickle.load(open("H_y.function","rb"))
    H_z = pickle.load(open("H_z.function","rb"))
    return H_x, H_y, H_z


def getB_x_t(path, B_x):

    x, y, z = path

    sets = np.multiply(1000,np.array((x,y,z)).T) # convert units to mm

    #print(sets)
    B_x_t = B_x(sets)

    endTime = time.perf_counter()


    return B_x_t


def getB_y_t(path, B_y):

    x, y, z = path

    sets = np.multiply(1000,np.array((x,y,z)).T)

    #print(sets)
    B_y_t = B_y(sets)
    
    endTime = time.perf_counter()


    return B_y_t
def getB_z_t(path, B_y):

    x, y, z = path

    sets = np.multiply(1000,np.array((x,y,z)).T)

    #print(sets)
    B_z_t = B_z(sets)
    
    endTime = time.perf_counter()



    return B_z_t



NUM_POINTS= int(100)


particleGenerator(NUM_POINTS)

particles = getParticles()
#print(particles)

filteredParticles = filter(particles)
print(filteredParticles)

trajectories = positionOverTime(filteredParticles,num_steps=500)
#print(trajectories)
particle0Trajectory = trajectories[0]

#oracleFunctionGenerator()




H_x, H_y, H_z = getOracleFunctions()

for traj in trajectories:
    B_x_t = getB_x_t(traj,H_x)
    B_y_t = getB_y_t(traj,H_y)
    B_z_t = getB_y_t(traj,H_z)

    axial_length = 0.30 # m
    T = axial_length / particles[0,0]
    t = np.linspace(0, T, num=len(B_x_t))



    #plt.plot(t,B_x_t)
    plt.plot(t,B_y_t)
    #plt.plot(t,B_z_t)
plt.show()



