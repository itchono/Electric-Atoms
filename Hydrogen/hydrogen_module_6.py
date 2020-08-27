# Hydrogen 6
# Hamiltonian Propagator

import numpy as np
from scipy import linalg as lg
from scipy.integrate import solve_ivp


def rho(p0, H_array, dt):
    '''
    Gets an array of all p(t), given an input set of Hamiltonian matrices, and a timestep dt

    Works best for small dt
    '''
    p = np.zeros((H_array.shape[0], *p0.shape)).astype(complex)

    p[0,:,:] = p0

    for i in range(1, H_array.shape[0]):

        p[i,:,:] = np.matmul(lg.expm(-1j*H_array[i-1,:,:]*dt), p[i-1,:,:])

        # Based on: June 22 Notes

    return p

def unitary(H_array, dt):
    '''
    Constructs the unitary transformation matrix for each point in a particle's journey based on the hamiltonian
    '''

    U = np.zeros((H_array.shape[0], *p0.shape)).astype(complex)

    U[0,:,:] = np.eye(H_array.shape[1])

    for i in range(1, H_array.shape[0]):

        U[i,:,:] = np.matmul(lg.expm(-1j*H_array[i-1,:,:]*dt), U[i-1,:,:])
        # Based on: June 22 Notes
    return U

## Construct differential equations?

def sai(sai_0, H, times):
    '''
    Produces the solution for sai(t), using pure vectorized form
    '''

    dt = times[1]-times[0]

    def equation_system(t, r):
        return np.matmul(H[t/dt,:,:], r)


    solution = solve_ivp(equation_system, times, sai_0)