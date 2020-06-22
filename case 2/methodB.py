import numpy as np
import scipy
import matplotlib.pyplot as plt

W = 10
w_0 = 5
w = 5


def p(x, t):

    p_00 = x[0]
    p_01_I = x[1]
    p_01_R = x[2]

    dp0dt = -2*p_01_I*W*np.cos(w*t)
    dpi1dt = w_0*p_01_R + (2*p_00-1) * W *np.cos(w*t)
    dpr1dt = -w_0*p_01_I

    return [dp0dt, dpi1dt, dpr1dt]
    
t = np.linspace(0, 1, 10000)
start = np.array([1, 0, 0])

solution = scipy.integrate.odeint(p, start, t)

rho_00 = solution[:,0]
rho_01 = solution[:,1] + 1j*solution[:,2]
rho_11 = 1-rho_00
fig, ax = plt.subplots()
ax.plot(t,rho_00,lw=2,label=r"$\rho_{00}$")
ax.plot(t,rho_01.imag,lw=2,label=r"$\mathfrak{Re}\rho_{01}$")
ax.set_xlabel(r"time, $t$")
ax.set_ylabel(r"density matrix elements, $\rho_{ij}(t)$")
ax.legend()
plt.show()




