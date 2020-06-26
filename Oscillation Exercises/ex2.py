import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# units in MHz
TRANSITON_FREQUENCY = 20
OSCILLATION_FREQUENCY = 10
RABI_FREQUENCY = 2


def solve(TRANSITON_FREQUENCY, OSCILLATION_FREQUENCY, RABI_FREQUENCY):
    ## density matrix (much faster)
    def equation_system(r,t,Omega,w0,w):
        rho_00, rho_01_r, rho_01_i = r

        rhodot_00 = -2*rho_01_i * Omega*cos(w*t)

        rhodot_01_r = -w0*rho_01_i

        rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)

        return rhodot_00, rhodot_01_r, rhodot_01_i


    # solution
    t = np.linspace(0,1,500) # time units in terms of microseconds

    r_init = np.array([0,0,0]) # initial starting state of the DEs

    w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*OSCILLATION_FREQUENCY # forced oscillation frequency vs energy level frequency
    Omega = 2*pi*RABI_FREQUENCY

    solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

    rho_11 = 1-solution[:,0][-1]

    print("Done")

    return rho_11

w = np.linspace(100, 250, 400)

prob = []

for i in w:
    prob.append(solve(177, i, 10))

print(prob)

plt.plot(w, np.array(prob))
plt.show()

