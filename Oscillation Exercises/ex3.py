import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


# units in MHz
TRANSITON_FREQUENCY = 177
OSCILLATION_FREQUENCY = 150
RABI_FREQUENCY = 10

def solve(TRANSITON_FREQUENCY, OSCILLATION_FREQUENCY, RABI_FREQUENCY):
    ## density matrix (much faster)
    def equation_system(r,t,Omega,w0,w):
        rho_00, rho_01_r, rho_01_i = r

        rhodot_00 = -2*rho_01_i * Omega*cos(w*t)

        rhodot_01_r = -w0*rho_01_i

        rhodot_01_i = +w0*rho_01_r + (2*rho_00 - 1) * Omega*cos(w*t)

        return rhodot_00, rhodot_01_r, rhodot_01_i


    # solution
    t = np.linspace(0,0.05,500) # time units in terms of microseconds

    r_init = np.array([0,0,0]) # initial starting state of the DEs


    w,w0 = 2*pi*TRANSITON_FREQUENCY,2*pi*OSCILLATION_FREQUENCY # forced oscillation frequency vs energy level frequency
    Omega = 2*pi*RABI_FREQUENCY


    solution = odeint(equation_system, r_init, t, args=(Omega,w,w0))

    b_z = 2*solution[:,0] - 1
    b_x = 2*solution[:,1]
    b_y = -2*solution[:,2]


    return b_x, b_y, b_z

bx, by, bz = solve(TRANSITON_FREQUENCY, OSCILLATION_FREQUENCY, RABI_FREQUENCY)

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlabel("$b_x$")
ax.set_ylabel("$b_y$")
ax.set_zlabel("$b_z$")

tick_spacing = 0.5

for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
    axis.set_major_locator(ticker.MultipleLocator(tick_spacing))

theta, phi = np.linspace(0,pi,20), np.linspace(0,2*pi,20)
Theta,Phi = np.meshgrid(theta,phi)

x = np.outer(np.cos(phi), np.sin(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.ones(np.size(phi)), np.cos(theta))

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
    ax.clear()
    north_blob, = ax.plot3D([0],[0],[1],'ro')
    south_blob, = ax.plot3D([0],[0],[-1],'go')
    bx, by, bz = solve(TRANSITON_FREQUENCY-100 + i, OSCILLATION_FREQUENCY, RABI_FREQUENCY)

    ax.text(0,0,0,f"{i}")
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    line, = ax.plot3D(bx,by,bz,lw=2,c=[i/200, 1-i/200, 1-i/200])

    

    return (line,) + (north_blob,) + (south_blob,)

ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=200, interval=5, blit=True)

plt.show()