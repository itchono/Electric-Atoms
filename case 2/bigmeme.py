import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
'''def H(t,Omega=10,w=100.0,w0=100):
    H = np.matrix([ [0,Omega*cos(w*t)],[Omega*cos(w*t),w0] ])
    return H
## exponentiating matrix
U = np.matrix(np.eye(2))
U_list = []
t = np.linspace(0,1,10000)
dt = t[1]-t[0]
for ti in t:
    U = lg.expm(-1j*H(ti,w=90)*dt) * U
    U_list.append(U)
p0 = [np.abs(U[0,0])**2 for U in U_list]    # probability to be in |0> at time t, given that you started in |0> at t=0
p1 = [np.abs(U[1,0])**2 for U in U_list]    # probability to be in |1> at time t, given that you started in |0> at t=0
fig, ax = plt.subplots()
ax.plot(t,p0,lw=2,label=r"$p_{0 \leftarrow 0}$")
ax.plot(t,p1,lw=2,label=r"$p_{1 \leftarrow 0}$")
ax.set_xlabel(r"time, $t$")
ax.set_ylabel(r"probabilities, $p(t)$")
ax.legend()
plt.show()'''
## density matrix 
def equation_system(r,t,Omega,w0,w):
    rho_00, rho_01_r, rho_01_i = r
    rhodot_00 = -2*rho_01_i * Omega*cos(w*t)
    rhodot_01_r = -w0*rho_01_i
    rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)
    return rhodot_00, rhodot_01_r, rhodot_01_i
# solution
t = np.linspace(0,1,1000)
r_init = np.array([1,0,0])
solution = odeint(equation_system,r_init,t,args=(30, 500, 500))
rho_00 = solution[:,0]
rho_01 = np.sqrt((solution[:,1] + 1j*solution[:,2])**2)
rho_11 = 1-rho_00
fig, ax = plt.subplots()
ax.plot(t,rho_00,lw=2,label=r"$\rho_{00}$")
ax.plot(t,rho_01.imag,lw=2,label=r"$\mathfrak{Re}\rho_{01}$")
ax.plot(t, 0.1*cos(200*t))
ax.set_xlabel(r"time, $t$")
ax.set_ylabel(r"density matrix elements, $\rho_{ij}(t)$")
ax.legend()
plt.show()