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

## rabi lineshape
w0 = 2*pi*177
T = 0.1
Omega = pi/T
frequencies = 2*pi*np.linspace(157,197,300)
rhos = []
t = np.linspace(0,T,1000)
r_init = np.array([1,0,0])
for w in frequencies:
    solution = odeint(equation_system,r_init,t,args=(Omega,w,w0))
    rho_00 = solution[:,0]
    rho_11_at_T = 1-rho_00[-1]
    rhos.append(rho_11_at_T)
rhos = np.array(rhos)
fig, ax = plt.subplots()
ax.plot(frequencies/(2*pi),rhos,'x',lw=2,label=r"$\rho_{00}$")
ax.set_ylabel(r"$\rho_{00}$")
ax.set_xlabel(r"$\omega/2\pi$")
ax.margins(0,0.1)
plt.show()

