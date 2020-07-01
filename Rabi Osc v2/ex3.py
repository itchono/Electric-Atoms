import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


# units in MHz
OMEGA = 1
DELTA = 1

def solve(DELTA):
    # solution
    times = np.linspace(0, 0.05, 100) # time units in terms of microseconds

    p_init = np.array([1, 0])

    h = sqrt(OMEGA**2 + DELTA**2)/2

    b_x = []
    b_y = []
    b_z = []

    for t in times:
        U_t_cos = np.array([[cos(h*t), 0*t], [0*t, cos(h*t)]])
        U_t_sin = -1j * OMEGA / sqrt(OMEGA**2 + DELTA**2) * np.array([[0,sin(h*t)], [sin(h*t),0]]) + -1j * DELTA / sqrt(OMEGA**2 + DELTA**2) * np.array([[sin(h*t), 0], [0, -sin(h*t)]])
        U_t = U_t_cos + U_t_sin

        p = np.matmul(U_t, p_init)

        b_z.append(np.abs(p[0])**2 + np.abs(p[1])**2)
        b_x.append(2*(p[0]*np.conjugate(p[1])).real)
        b_y.append(2*(p[0]*np.conjugate(p[1])).imag)

    return b_x, b_y, b_z

bx, by, bz = solve(1)

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
    bx, by, bz = solve(i-50)
    ax.text(0,0,0,f"{i}")
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    line, = ax.plot3D(bx,by,bz,lw=2,c=[i/200, 1-i/200, 1-i/200])

    return (line,) + (north_blob,) + (south_blob,)

ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=200, interval=5, blit=True)

plt.show()