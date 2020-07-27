# Hydrogen 6
# Hamiltonian Propagator

import numpy as np
from scipy import linalg as lg


def rho(p0, H_array, dt):
    '''
    Gets an array of all p(t), given an input set of Hamiltonian matrices, and a timestep dt

    Works best for small dt
    '''
    p = np.zeros((H_array.shape[0], *p0.shape))

    p[0,:,:] = p0

    for i in range(1, H_array.shape[0]):
        p[i,:,:] = lg.expm(-1j*H_array[i-1,:,:]*dt) * p[i-1,:,:]
        # Based on: June 22 Notes

    return p

def unitary(H_array, dt):
    '''
    Constructs the unitary transformation matrix for each point in a particle's journey based on the hamiltonian
    '''

    U = np.zeros((H_array.shape[0], *p0.shape))

    U[0,:,:] = np.eye(H_array.shape[1])

    for i in range(1, H_array.shape[0]):
        U[i,:,:] = lg.expm(-1j*H_array[i-1,:,:]*dt) * U[i-1,:,:]
        # Based on: June 22 Notes
    return p

## Construct differential equations?