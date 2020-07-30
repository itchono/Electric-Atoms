import numpy as np 
import pickle
import time
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi,cos,sin
import scipy.linalg as lg
from scipy.constants import e,c,h,hbar,u,m_e,epsilon_0
muB = 2*pi* e*hbar/(2*m_e)/(h*1e10)                     # 2pi* MHz/G
ea0 = 2*pi* 4*pi*epsilon_0*hbar**2/(e * m_e)/(h*1e4)    # 2pi* MHz/(V/cm)
from sympy.physics.wigner import wigner_3j,wigner_6j
import copy


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

    T = slowestTime(particles, axial_length)

    times = np.tile(np.linspace(0, T, num=num_steps), (particles.shape[0], 1)).T # make range of times, and stack them up to work on the array

    x = vx * times
    y = y0 + vy * times
    z = z0 + vz * times
    # apply newton's law

    return np.transpose(np.array((x, y, z)), [2, 0, 1])

def slowestTime(particles, axial_length = 0.30):
    '''
    Get Time taken to traverse the x distance, on the SLOWEST particle [timescale used by every other calculation]
    '''
    vx = particles[:,0]
    return axial_length / np.amin(vx[np.nonzero(vx)])

def magnetic_fields(T, omega=10**6, driving_frequency = 177*10**6, num_steps = 500):
    '''
    Placeholder to Ryan's magnetic field thing.
    '''
    Bz0 = omega # <-- omega value which is to be changed

    times = np.linspace(0, T, num=num_steps)

    b = np.zeros((num_steps, 3))

    b[:,2] = Bz0 * cos(2*pi*driving_frequency * times)

    return b

def hamiltonian(b_fields, H0, Mx, My, Mz):
    '''
    Produces the complete hamiltonian for each particle in the input array
    '''
    H_array = np.zeros((*b_fields.shape[0:2], *H0.shape)).astype(complex)

    for i in range(len(b_fields)):
        H_int = magnetic_interaction(b_fields[i,:,0], Mx) + magnetic_interaction(b_fields[i,:,1], My) + magnetic_interaction(b_fields[i,:,2], Mz)
        H_array[i,:,:,:] = (H_int + H0)
        # TODO is there a faster way to do this

    return H_array

def magnetic_interaction(b_field, mu):
    '''
    Produces H(t), given an input B(t) of one axis, with T timesteps

    H(t) is a T x 4 x 4 matrix (can be reconfigured)
    B(t) is a T x 1 array

    This needs to be done with each of the Bx, By, Bz directions.
    '''
    return -np.array([mu*b for b in b_field]) # TODO is there a faster way to do this in numpy

def save_matrices(configuration_name):
    '''
    Saves H0 and mu matrices to file for later use.

    Should reconfigure whenever using different atomic values.
    '''

    ## atomic functions (originally by Amar)
    # hbar = 1
    # All energies in 2*pi*MHz
    # All times in us
    # Electric fields in V/cm
    class AtomicState():
        def __init__(self,energy=0,L=0,J=1/2,F=0):
            self.energy = energy       
            self.S = 1/2    # Intrinsic angular momentum of electron
            self.I = 1/2	# Intrinsic angular momentum of the hydrogen nucleus
            self.L = L      # Orbital angular momentum of electron
            self.J = J      # Total angular momentum of electron
            self.F = F
            self.mF = F
            self.P = (-1)**self.L      # parity

        def __repr__(self):
            attribs = [str(s) for s in [self.S,self.L,self.J]]
            string = ','.join(attribs)

            return f"|L = {self.L}, J = {self.J}; F,mF = {self.F},{self.mF}>"

    def sublevel_expand(basis):
        # makes all the m_F sublevels given one stretched (mF = F) level
        newbasis = []
        for ket in basis:
            for mF in np.arange(-ket.F,ket.F+1,1):
                newket = copy.deepcopy(ket)
                newket.mF = mF				
                newbasis.append(newket)
        return newbasis

    ## Definition of dipole matrix elements & reduced matrix elements
    def delta(i,j): return (i==j)*1.0

    def M1_moment(A,B,q=0):
        """M1 matrix element: <A|mu|B>
            Units are mu_B """
        if (A.P*B.P == +1):	# check to see if the two states have same parity
            return (-1)**(A.F-A.mF) * wigner_3j(A.F,1,B.F,-A.mF,q,B.mF) * M1_reduced_F(A,B)
        else: return 0

    def M1_reduced_F(A,B,gI=1.521032e-3):
        """F-reduced matrix element <F|| mu ||F'>"""
        # WOAH NOTE: what is this?
        # Proton magnetic moment, gI = 0.001521032 Bohr magnetons
        return np.sqrt((2*A.F+1)*(2*B.F+1)) * ( (-1)**(A.J+A.I+B.F+1) * delta(A.I,B.I) * wigner_6j(A.F,1,B.F,B.J,A.I,A.J) * M1_reduced_J(A,B)  + (-1)**(B.J+B.I+A.F+1) * delta(A.J,B.J) * wigner_6j(A.F,1,B.F,B.I,A.J,A.I) * gI * np.sqrt(A.I*(A.I+1)*(2*A.I+1)) )

    def M1_reduced_J(A,B,gL=-0.9995,gS=-2.0023193):
        """J-reduced matrix element <J|| mu ||J'>"""
        if (A.L==B.L) and (A.S==B.S):
            return np.sqrt((2*A.J+1)*(2*B.J+1)) * ( (-1)**(A.L+A.S+B.J+1) * wigner_6j(A.J,1,B.J,B.L,A.S,A.L) * gL * np.sqrt(A.L*(A.L+1)*(2*A.L+1)) + (-1)**(B.L+B.S+A.J+1) * wigner_6j(A.J,1,B.J,B.S,A.L,A.S) * gS * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) ) 
        else: return 0

        
    ############# Atomic data ################################
    # 2S_1/2 states

    # SELF NOTE: 2pi*(59.2+909.872) MHz is the distance between 2p1/2 F = 0 and 2s1/2 F = 0

    ground_state = AtomicState(energy=0,L=0,J=1/2,F=0)
    excited_state = AtomicState(energy=2*pi*177,L=0,J=1/2,F=1)

    # FIRST STEP: define our basis
    basis = sublevel_expand([ground_state, excited_state])

    N = len(basis)

    # STEP 2: Steady State Hamiltonian
    H0 = np.matrix(np.diag([b.energy for b in basis]))


    ## Operator matrices 
    Mz = np.matrix(np.zeros((N,N)))
    Mplus = np.matrix(np.zeros((N,N)))
    Mminus = np.matrix(np.zeros((N,N)))

    # TODO: Refactor using meshgrids to make things faster
    # Need to break down atomic state object into np vector

    for i in range(N):
        for j in range(N):		
            mz = M1_moment(basis[i],basis[j],q=0)
            mplus = M1_moment(basis[i],basis[j],q=+1)
            mminus = M1_moment(basis[i],basis[j],q=-1)	
            Mz[i,j],Mplus[i,j],Mminus[i,j] = mz,mplus,mminus
    Mx = (Mminus - Mplus)/np.sqrt(2)
    My = (Mminus + Mplus)/(1j*np.sqrt(2))

    np.save(f"{configuration_name}_H0.npy", H0)
    np.save(f"{configuration_name}_z.npy", Mz)
    np.save(f"{configuration_name}_x.npy", Mx)
    np.save(f"{configuration_name}_y.npy", My)

def load_matrices(configuration_name):
    '''
    Loads the H0, Mx, My, Mz matrices from a configuration name
    '''
    try:
        H0 = np.load(f"{configuration_name}_H0.npy")
        Mx = np.load(f"{configuration_name}_x.npy")
        My = np.load(f"{configuration_name}_y.npy")
        Mz = np.load(f"{configuration_name}_z.npy")
        return (H0, Mx, My, Mz)
    except:
        print(f"Matrices with configuration name {configuration_name} could not be found.")
        return None

def rho(p0, H_array, dt):
    '''
    Gets an array of all p(t), given an input set of Hamiltonian matrices, and a timestep dt

    Works best for small dt
    '''
    p = np.zeros((H_array.shape[0], *p0.shape)).astype(complex)

    p[0,:,:] = p0

    for i in range(1, H_array.shape[0]):
        p[i,:,:] = lg.expm(-1j*H_array[i-1,:,:]*dt) * p[i-1,:,:]
        # Based on: June 22 Notes

    return p

def unitary(H_array, dt):
    '''
    Constructs the unitary transformation matrix for each point in a particle's journey based on the hamiltonian
    '''

    U = np.zeros((H_array.shape[0], *p0.shape)).astype(complex)

    U[0,:,:] = np.eye(H_array.shape[1])

    for i in range(1, H_array.shape[0]):
        U[i,:,:] = lg.expm(-1j*H_array[i-1,:,:]*dt) * U[i-1,:,:]
        # Based on: June 22 Notes
    return U


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
    #cProfile.run("profiler()")

    # Requires as prerequisite code (only run once)
    '''
    NUM_POINTS= int(20000)
    particleGenerator(NUM_POINTS)
    save_matrices("hydrogen_matrix")
    '''
    particles = getParticles()
    filteredParticles = filter(particles)
    T = slowestTime(filteredParticles)
    
    b = np.array([magnetic_fields(T, num_steps=int(1e4))])

    H0, Mx, My, Mz = load_matrices("hydrogen_matrix")
    H = hamiltonian(b, H0, Mx, My, Mz)

    print(H[0, 0,:,:])

    p0 = np.diag([1,0,0,0])

    print(p0)

    U = unitary(H[0], T/1e4) # only 

    p_00 = np.abs(U[:,0,0])**2

    times = np.linspace(0, T, num=int(1e4))

    plt.plot(times, p_00, 'x')

    plt.show()






