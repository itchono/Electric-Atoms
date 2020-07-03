import numpy as np
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# Rotating Wave Approximation; Unitary Transformation Matrix for system at time T
def U(t,Omega,w0,w):
    h = np.sqrt(Omega**2 + (w-w0)**2)/2
    return np.matrix([[cos(h*t),0],[0,cos(h*t)]]) + \
            -1j*(Omega/(2*h)) * np.matrix([[0,sin(h*t)],[sin(h*t),0]]) + \
            -1j*((w-w0)/(2*h)) * np.matrix([[sin(h*t),0],[0,-sin(h*t)]]) 


w0 = 2*pi*177
T = 0.1
Omega = pi/2/T
t = np.linspace(0,T,50)

fig = plt.figure()
ax = p3.Axes3D(fig)

ANIM_RANGE = 30 * 6

ax.set_xlabel("$b_x$")
ax.set_ylabel("$b_y$")
ax.set_zlabel("$b_z$")

def init():
    # Annotate points
    ax.text(0,0,1.2,r"$|1\rangle$")
    ax.text(0,0,-1.2,r"$|0\rangle$")

    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    north_blob, = ax.plot3D([0],[0],[1],'ro')
    south_blob, = ax.plot3D([0],[0],[-1],'go')

    return (north_blob,) + (south_blob,)

bruh1 = []
bruh2 = []
bruh3 = []

def animate(i):
    ax.clear()
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    w = w0-ANIM_RANGE+i
    
    north_blob, = ax.plot3D([0],[0],[1],'ro')
    south_blob, = ax.plot3D([0],[0],[-1],'go')
    vg = np.array([U(ti,Omega,w0,w)[0,0] for ti in t])
    ve = np.array([U(ti,Omega,w0,w)[1,0] for ti in t])

    bx = 2*(ve*vg.conjugate()).real
    by = 2*(ve*vg.conjugate()).imag
    bz = np.abs(ve)**2 - np.abs(vg)**2

    line, = ax.plot3D(bx,by,bz)
    terminal, = ax.plot3D([bx[-1]],[by[-1]],[bz[-1]], 'bo')

    bruh1.append(w)
    bruh2.append(bz[-1])
    bruh3.append(max(bz))

    mag = sqrt((Omega)**2 + (w0-w)**2)

    h_arrow, = ax.plot3D([0, Omega/mag], [0, 0], [0, (w0-w)/mag])

    return (line,) + (h_arrow,) + (north_blob,) + (south_blob,) + (terminal,)

ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=2*ANIM_RANGE, interval=1, blit=True)

plt.show()


fig, ax = plt.subplots()
ax.plot(np.array(bruh1)/(2*pi),bruh2,'x',label=r"$\rho_{00}$")
ax.plot(np.array(bruh1)/(2*pi),bruh3,'o',label=r"$\rho_{00}$")

# peak is 1/(1+(delta/omega)^2)
# periodicity is sqrt(1+(DELTA/BIG_OMEGA)**2)

ax.set_ylabel(r"$\rho_{00}$")
ax.set_xlabel(r"$\omega/2\pi$")
ax.margins(0,0.1)
plt.show()

fig, axes = plt.subplots(nrows=1)
RANGE = 30
DELTAS = np.linspace(-RANGE*6, RANGE*6, 70)
# predict analytic solution

prediction = []

T = 0.1
BIG_OMEGA = pi/T


for DELTA in DELTAS:
    amp_factor = 1/(1+(DELTA/BIG_OMEGA)**2)
    time_factor = sqrt(1+(DELTA/BIG_OMEGA)**2) # YES I DID IT
    prediction.append(amp_factor*(sin(time_factor*BIG_OMEGA/2*T))**2)
axes.plot(DELTAS/6,prediction,lw=2)

axes.margins(0,0.1)
plt.show()