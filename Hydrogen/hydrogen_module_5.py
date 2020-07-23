# Hydrogen 5
# Hamiltonian Generator

import numpy as np
from numpy import pi,cos,sin
import scipy.linalg as lg
from scipy.constants import e,c,h,hbar,u,m_e,epsilon_0
muB = 2*pi* e*hbar/(2*m_e)/(h*1e10)                     # 2pi* MHz/G
ea0 = 2*pi* 4*pi*epsilon_0*hbar**2/(e * m_e)/(h*1e4)    # 2pi* MHz/(V/cm)
from sympy.physics.wigner import wigner_3j,wigner_6j
import copy

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
    excited_state = AtomicState(energy=0,L=0,J=1/2,F=1) # Remember: tau is in microseconds --> 0.13 seconds

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

save_matrices("hydrogen_matrix")

b = np.array([[[1,2,3], [4,5,6]], [[2,3,4], [5,6,7]]])

H0, Mx, My, Mz = load_matrices("hydrogen_matrix")

H = hamiltonian(b, H0, Mx, My, Mz)

print(H)