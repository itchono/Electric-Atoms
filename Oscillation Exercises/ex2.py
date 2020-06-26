import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def equation_system(r,t,Omega,w0,w):
    rho_00, rho_01_r, rho_01_i = r

    rhodot_00 = -2*rho_01_i * Omega*cos(w*t)

    rhodot_01_r = -w0*rho_01_i

    rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)

    return rhodot_00, rhodot_01_r, rhodot_01_i

yeet = np.linspace(160, 200, 81)

prob = []

for i in yeet:

    TRANSITON_FREQUENCY = 177
    BIG_OMEGA = 1
    # solution
    t = np.linspace(0,np.pi/BIG_OMEGA,200) # time units in terms of microseconds

    r_init = np.array([1,0,0]) # initial starting state of the DEs

    w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*i # forced oscillation frequency vs energy level frequency
    Omega = 2*pi*BIG_OMEGA

    solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

    rho_11 = 1-solution[:,0][-1]
    print("Done")

    prob.append(rho_11)


plt.plot(yeet, np.array(prob), 'x',lw=2,label=r"$\rho_{00}$")
plt.show()

