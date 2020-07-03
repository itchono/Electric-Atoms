import numpy as np
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

ANIM_RANGE = 30 * 6
# Range of frequencies we want. multiplying by 6 is so that we multiply by roughly 2 pi to turn into angular frequency.

w0 = 2*pi*177 # transition energy for m = 0 2s hyperfine transition in hydrogen
T = 0.1
Omega = pi/T


# Rotating Wave Approximation; Unitary Transformation Matrix for system at time T
def U(t,Omega,w0,w):
    h = np.sqrt(Omega**2 + (w-w0)**2)/2
    return np.matrix([[cos(h*t),0],[0,cos(h*t)]]) + \
            -1j*(Omega/(2*h)) * np.matrix([[0,sin(h*t)],[sin(h*t),0]]) + \
            -1j*((w-w0)/(2*h)) * np.matrix([[sin(h*t),0],[0,-sin(h*t)]]) 

t = np.linspace(0,T,50) # set of all times being used for evaluation of the bloch vector's trajectory

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlim3d([-1.0, 1.0])
ax.set_ylim3d([-1.0, 1.0])
ax.set_zlim3d([-1.0, 1.0])

ax.set_xlabel("$b_x$")
ax.set_ylabel("$b_y$")
ax.set_zlabel("$b_z$")

ax.text(0,0,1.2,r"$|1\rangle$")
ax.text(0,0,-1.2,r"$|0\rangle$")

line, = ax.plot3D([],[],[], label = "$b_z(t)$ (trajectory)")
terminal, = ax.plot3D([],[],[], 'bo', label = "$b_z(T)$ (final)")
north_blob, = ax.plot3D([0],[0],[1],'ro')
south_blob, = ax.plot3D([0],[0],[-1],'go')
h_arrow, = ax.plot3D([], [], [], label="h vector")

recorded_delta = [] # recorded frequencies during animation
recorded_bz_final = [] # recorded bz(T) during animation
recorded_bz_max = [] # recorded max(bz) during animation

def animate(i):
    w = w0-ANIM_RANGE+i # driving frequency

    vg = np.array([U(ti,Omega,w0,w)[0,0] for ti in t])
    ve = np.array([U(ti,Omega,w0,w)[1,0] for ti in t])

    bx = 2*(ve*vg.conjugate()).real
    by = 2*(ve*vg.conjugate()).imag
    bz = np.abs(ve)**2 - np.abs(vg)**2

    line.set_data(bx,by)
    line.set_3d_properties(bz)

    terminal.set_data([bx[-1]],[by[-1]])
    terminal.set_3d_properties([bz[-1]])

    recorded_delta.append(w)
    recorded_bz_final.append(bz[-1])
    recorded_bz_max.append(max(bz))

    mag = sqrt((Omega)**2 + (w0-w)**2)

    h_arrow.set_data([0, Omega], [0, 0])
    h_arrow.set_3d_properties([0, (w0-w)])

    return (line,) + (h_arrow,) + (north_blob,) + (south_blob,) + (terminal,)

ani = animation.FuncAnimation(fig, animate,
                                frames=2*ANIM_RANGE, interval=1, blit=True)
plt.title("Bloch Sphere Trajectory vs Delta")
plt.legend()
plt.show()


fig, ax = plt.subplots()
ax.plot(np.array(recorded_delta)/(2*pi),recorded_bz_final,'x',label="$b_z(T)$")
ax.plot(np.array(recorded_delta)/(2*pi),recorded_bz_max,'o',label="$max({b_z(t)})$")
# peak is 1/(1+(delta/omega)^2)
# periodicity is sqrt(1+(DELTA/BIG_OMEGA)**2)
ax.set_xlabel(r"$\omega/2\pi$")
plt.title("$Recorded\ b_z(T)\ vs\ max({b_z(t)})$")
plt.legend()
plt.show()