import numpy as np


def hamiltonian(ang_freq, B_0, end_time, num_steps):
    '''
    Generates hamiltonian at all time steps
    '''
    H_0 = np.zeros((num_steps, 2, 2))
    H_0[:, 1, 1].fill(2*np.pi*177*10**6)

    H_int = np.zeros((num_steps, 2, 2))
    H_int[:, 0, 1] = B_0*2*np.pi*10**6*np.cos(2*np.pi*ang_freq*np.linspace(0, end_time, num_steps))
    H_int[:, 1, 0] = B_0*2*np.pi*10**6*np.cos(2*np.pi*ang_freq*np.linspace(0, end_time, num_steps))

    H = (H_0 + H_int).astype(complex) / (2*np.pi*177*10**6)

    return H

def propagate(starting_state, ang_freq, B, end_time, num_steps):
    '''
    Uses schrodinger equation to propagate over time
    '''
    result = np.zeros((2, num_steps)).astype(complex)

    result[:, 0] = starting_state
    H = hamiltonian(ang_freq, B, end_time, num_steps)

    dt = complex(end_time / num_steps) # timestep

    for i in range(num_steps-1):

        dr = np.matmul(H[i,:,:],result[:,i])/(0+1j) * dt
        result[:,i+1] = result[:,i] + dr
    return result


if __name__ == "__main__":
    P_0 = np.array([1,0]) # Initial Starting State

    P_t = propagate(P_0, 0, 0.01, 10**-4, 5)
    # DC mode; no transitions are driven, as expected

    print(P_t[:, -1]**2)


    P_t = propagate(P_0, 177000000, 10, 10**-4, 50)
    # AC; ????

    print(P_t[:, -1]**2)

    