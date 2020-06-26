import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# units in MHz
TRANSITON_FREQUENCY = 177
OSCILLATION_FREQUENCY = 170
RABI_FREQUENCY = 10


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

rho_00 = solution[:,0]

rho_01 = solution[:,1] + 1j*solution[:,2]

rho_11 = 1-rho_00

maxp = max(rho_00)


fig, axes = plt.subplots(nrows=3)
axes[0].plot(t,rho_00,lw=2,label=r"$\rho_{00}$",color='C0')
axes[0].set_ylabel(r"$\rho_{00}$",color='C0')
axes[1].plot(t,rho_01.imag,lw=2,label=r"$\mathfrak{Im} \, \rho_{01}$",color='C1')
axes[1].set_ylabel(r"$\mathfrak{Im} \rho_{01}$",color='C1')
axes[2].plot(t,cos(w*t),lw=2,alpha=0.7,label=r"$\mathcal{B}_z$",color='C2')
axes[2].set_ylabel(r"$\mathcal{B}_z$",color='C2')
ax = axes[1].twinx()
axes[1].set_xlabel(r"time, $t$ (microseconds)")
axes[0].margins(0,0.1)
axes[1].margins(0,0.1)
axes[2].margins(0,0.1)

print(f"w/w0: {OSCILLATION_FREQUENCY/TRANSITON_FREQUENCY}\nRabi: {RABI_FREQUENCY}")
print(f"max P: {maxp}")
plt.show()

