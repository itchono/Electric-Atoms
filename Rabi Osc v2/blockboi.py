import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import odeint
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

## analytic solution (with RWA)
def U(t,Omega,w0,w):
    h = np.sqrt(Omega**2 + (w-w0)**2)/2
    return np.matrix([[cos(h*t),0],[0,cos(h*t)]]) + \
            -1j*(Omega/(2*h)) * np.matrix([[0,sin(h*t)],[sin(h*t),0]]) + \
            -1j*((w-w0)/(2*h)) * np.matrix([[sin(h*t),0],[0,-sin(h*t)]]) 
# solution
w0 = 2*pi*177
T = 0.1
Omega = pi/T
t = np.linspace(0,T,50)

fig = plt.figure()
ax = p3.Axes3D(fig)

ANIM_RANGE = 30

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

def animate(i):
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    
    north_blob, = ax.plot3D([0],[0],[1],'ro')
    south_blob, = ax.plot3D([0],[0],[-1],'go')
    vg = np.array([U(ti,Omega,w0,w0-ANIM_RANGE+i)[0,0] for ti in t])
    ve = np.array([U(ti,Omega,w0,w0-ANIM_RANGE+i)[1,0] for ti in t])

    bx = 2*(ve*vg.conjugate()).real
    by = 2*(ve*vg.conjugate()).imag
    bz = np.abs(ve)**2 - np.abs(vg)**2

    line, = ax.plot3D(bx,by,bz,lw=2,c=[i/(2*ANIM_RANGE), 1-i/(2*ANIM_RANGE), 1-i/(2*ANIM_RANGE)])

    print(bz[-1])

    mag = sqrt((Omega)**2 + (ANIM_RANGE-i)**2)

    h_arrow, = ax.plot3D([0, Omega/mag], [0, 0], [0, (ANIM_RANGE-i)/mag], lw=2,c=[i/(2*ANIM_RANGE), 1-i/(2*ANIM_RANGE), 1-i/(2*ANIM_RANGE)])

    return (line,) + (h_arrow,) + (north_blob,) + (south_blob,)

ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=2*ANIM_RANGE, interval=1, blit=True)

plt.show()