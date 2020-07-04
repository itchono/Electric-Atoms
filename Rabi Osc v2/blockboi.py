import numpy as np
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

RANGE = 50 * 6
# Range of frequencies we want. multiplying by 6 is so that we multiply by roughly 2 pi to turn into angular frequency.

w0 = 2*pi*177 # transition energy for m = 0 2s hyperfine transition in hydrogen
T = 0.1
Omega = pi/T # PI pulse; change to other durations for cool effects.

# Rotating Wave Approximation; Unitary Transformation Matrix for system at time T
def U(t,Omega,w0,w):
    h = np.sqrt(Omega**2 + (w-w0)**2)/2
    return np.matrix([[cos(h*t),0],[0,cos(h*t)]]) + \
            -1j*(Omega/(2*h)) * np.matrix([[0,sin(h*t)],[sin(h*t),0]]) + \
            -1j*((w-w0)/(2*h)) * np.matrix([[sin(h*t),0],[0,-sin(h*t)]]) 

t = np.linspace(0,T,50) # set of all times being used for evaluation of the bloch vector's trajectory

fig = plt.figure(figsize=plt.figaspect(0.4))

ax = fig.add_subplot(122, projection='3d')

ax.set_xlim3d([-1.0, 1.0])
ax.set_ylim3d([-1.0, 1.0])
ax.set_zlim3d([-1.0, 1.0])
ax.set_xlabel("$b_x$")
ax.set_ylabel("$b_y$")
ax.set_zlabel("$b_z$")

ax.text(0,0,1.2,r"$|1\rangle$")
ax.text(0,0,-1.2,r"$|0\rangle$")
# label ground and excited states

line, = ax.plot3D([],[],[], label = "$b(t)$ (trajectory)")
terminal, = ax.plot3D([],[],[], 'bo', label = "$b(T)$ (final)")
north_blob, = ax.plot3D([0],[0],[1],'ro')
south_blob, = ax.plot3D([0],[0],[-1],'go')
h_arrow, = ax.plot3D([], [], [], label="$2h = \Omega*x + \Delta*z$", linewidth=2)

ax.set_title("Bloch Sphere", loc="left")
ax.legend()

ax2 = fig.add_subplot(121)
ax2.set_xlabel(r"$\omega/2\pi$")

recorded_delta = [] # recorded frequencies during animation
recorded_bz_final = [] # recorded bz(T) during animation
recorded_bz_max = [] # recorded max(bz) during animation

bzarr, = ax2.plot([0],[0],'-',label="$b_z(T)$")
bzmaxarr, = ax2.plot([0],[0],'-',label="$max({b_z(t)})$")

ax2.set_xlim([(w0-RANGE)/(2*pi), (w0+RANGE)/(2*pi)])
ax2.set_ylim([-1, 1])

ax2.margins(0,0.1)
ax2.set_title("$Recorded\ b_z(T)\ vs\ max({b_z(t)})$")
ax2.legend()

def animate(i):
    '''
    Update the animation
    '''
    w = w0 - RANGE + i # set the new driving frequency

    arr = [U(ti,Omega,w0,w) for ti in t]

    vg = np.array([el[0, 0] for el in arr])
    ve = np.array([el[1, 0] for el in arr])

    bx = 2*(ve*vg.conjugate()).real
    by = 2*(ve*vg.conjugate()).imag
    bz = np.abs(ve)**2 - np.abs(vg)**2
    # evaluate bloch vector components over time

    line.set_data(bx,by)
    line.set_3d_properties(bz)

    terminal.set_data([bx[-1]],[by[-1]])
    terminal.set_3d_properties([bz[-1]])

    if len(recorded_delta) < 2*RANGE:
        recorded_delta.append(w)
        recorded_bz_final.append(bz[-1])
        recorded_bz_max.append(max(bz))

    h_arrow.set_data([0, Omega/RANGE], [0, 0])
    h_arrow.set_3d_properties([0, (w0-w)/(RANGE)])

    bzarr.set_data(np.array(recorded_delta)/(2*pi),recorded_bz_final)
    bzmaxarr.set_data(np.array(recorded_delta)/(2*pi),recorded_bz_max)

    return (line,) + (h_arrow,) + (north_blob,) + (south_blob,) + (terminal,) + (bzmaxarr,) + (bzarr,)

ani = animation.FuncAnimation(fig, animate,frames=2*RANGE, interval=1, blit=True)
# set blit to False if you want to rotate the plot (warning: slow)
plt.show()