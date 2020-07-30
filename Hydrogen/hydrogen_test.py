import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def H(t,Omega=10,w=100.0,w0=100):
    H = np.matrix([ [0,Omega*cos(w*t)],[Omega*cos(w*t),w0] ])
    return H
## exponentiating matrix (slow)
U = np.matrix(np.eye(2))
U_list = []
t = np.linspace(0,1,10000)
dt = t[1]-t[0]
for ti in t:
    U = lg.expm(-1j*H(ti,w=100)*dt) * U
    print(lg.expm(-1j*H(ti,w=100)*dt))
    U_list.append(U)
p0 = [np.abs(U[0,0])**2 for U in U_list]    # probability to be in |0> at time t, given that you started in |0> at t=0
p1 = [np.abs(U[1,0])**2 for U in U_list]    # probability to be in |1> at time t, given that you started in |0> at t=0
fig, ax = plt.subplots()
ax.plot(t,p0,lw=2,label=r"$p_{0 \leftarrow 0}$")
ax.plot(t,p1,lw=2,label=r"$p_{1 \leftarrow 0}$")
ax.set_xlabel(r"time, $t$")
ax.set_ylabel(r"probabilities, $p(t)$")
ax.legend()
plt.show()
