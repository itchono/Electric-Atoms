import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# units in MHz
TRANSITON_FREQUENCY = 177
BIG_OMEGA = 1

# BIG OMEGA IS NOT RABI FREQUENCY


## density matrix (much faster)
def equation_system(r,t,Omega,w0,w):
    rho_00, rho_01_r, rho_01_i = r

    rhodot_00 = -2*rho_01_i * Omega*cos(w*t)

    rhodot_01_r = -w0*rho_01_i

    rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)

    return rhodot_00, rhodot_01_r, rhodot_01_i


# solution
t = np.linspace(0,1/BIG_OMEGA,1000) # time units in terms of microseconds

r_init = np.array([1,0,0]) # initial starting state of the DEs


w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*177 # forced oscillation frequency vs energy level frequency
Omega = 2*pi*BIG_OMEGA


solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

rho_00 = solution[:,0]

rho_11 = 1-rho_00

maxp = max(rho_00)


fig, axes = plt.subplots(nrows=1)
axes.plot(t,rho_11,lw=2,label=r"$\rho_{11}$",color='C0')
axes.set_ylabel(r"$\rho_{11}$",color='C0')



r_init = np.array([1,0,0]) # initial starting state of the DEs

w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*178 # forced oscillation frequency vs energy level frequency
Omega = 2*pi*BIG_OMEGA

solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

rho_00 = solution[:,0]
rho_11 = 1-rho_00

axes.plot(t,rho_11,lw=2,label=r"$\rho_{11}$",color='C1')
axes.set_ylabel(r"$\rho_{11}$",color='C0')
axes.margins(0,0.1)

r_init = np.array([1,0,0]) # initial starting state of the DEs

w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*176 # forced oscillation frequency vs energy level frequency
Omega = 2*pi*BIG_OMEGA

solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

rho_00 = solution[:,0]
rho_11 = 1-rho_00

axes.plot(t,rho_11,lw=2,label=r"$\rho_{11}$",color='C2')
axes.set_ylabel(r"$\rho_{11}$",color='C0')
axes.margins(0,0.1)

plt.show()

