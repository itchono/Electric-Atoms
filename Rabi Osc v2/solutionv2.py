import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt, e
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# units in MHz
TRANSITON_FREQUENCY = 177
BIG_OMEGA = 2

DELTA = 3

# BIG OMEGA IS NOT RABI FREQUENCY


## density matrix (much faster)
def equation_system(r,t,Omega,w0,w):
    rho_00, rho_01_r, rho_01_i = r

    rhodot_00 = -2*rho_01_i * Omega*cos(w*t)

    rhodot_01_r = -w0*rho_01_i

    rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)

    return rhodot_00, rhodot_01_r, rhodot_01_i


# solution
t = np.linspace(0,1/BIG_OMEGA,2000) # time units in terms of microseconds

r_init = np.array([1,0,0]) # initial starting state of the DEs

w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*177 # forced oscillation frequency vs energy level frequency
Omega = 2*pi*BIG_OMEGA

solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))
fig, axes = plt.subplots(nrows=1)
axes.plot(t,1-solution[:,0],lw=2,label=r"$\rho_{11}$",color='C0')

w0 = 2*pi*(177+DELTA) # forced oscillation frequency vs energy level frequency
solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))
axes.plot(t,1-solution[:,0],lw=2,label=f"+{DELTA}",color='C1')

w0 = 2*pi*(177-DELTA) # forced oscillation frequency vs energy level frequency
solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))
axes.plot(t,1-solution[:,0],lw=2,label=f"-{DELTA}",color='C2')


# predict analytic solution
amp_factor = 1/(1+(DELTA/BIG_OMEGA)**2)
time_factor = sqrt(1+(DELTA/BIG_OMEGA)**2) # YES I DID IT

prediction = amp_factor*(sin(2*pi*time_factor*BIG_OMEGA/2*t))**2
axes.plot(t,prediction, "--",lw=2,label="prediction",color='C3')

axes.margins(0,0.1)
plt.legend()
plt.show()

