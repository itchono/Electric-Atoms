# Hydrogen 6
# Hamiltonian Propagator

import numpy as np
from scipy import linalg as lg


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

## Construct differential equations?

def equation_system(r,t,Omega,w0,w):
    # Need 9 DEs

    # Problem: can't solve magnetic fields analytically <- uh oh

    rho_0011, rho_1122

    rho_00, rho_10_r, rho_10_i = r
    rhodot_00 = 2*rho_10_i * Omega*cos(w*t)
    rhodot_10_r = w0*rho_10_i
    rhodot_10_i = -w0*rho_10_r - (2*rho_00 - 1) * Omega*cos(w*t)
    return rhodot_00, rhodot_10_r, rhodot_10_i


solution = odeint(equation_system,r_init,t,args=(Omega,w,w0))