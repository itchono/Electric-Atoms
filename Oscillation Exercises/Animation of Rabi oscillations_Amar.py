# Animation of Rabi oscillations
# version 1, Amar Vutha, 2015-11-10
# Notes: 
#       Feel free to modify and improve this program!
#       If you improve it significantly, please send me a copy so that I can use it for future courses.
#
import numpy as np
from numpy import sin, cos, pi, sqrt
from scipy import linalg as lg
from scipy.misc import factorial
from scipy.constants import c,e,h,hbar,u,pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
​
​
## operators
​
def H_int(t,Omega=1,w=10.0,w0=10):
    Hint = np.matrix([ [(w-w0)/2,Omega/2],[Omega/2,-(w-w0)/2] ])
    return Hint
    
def bloch_vector(psi):
    b_x = psi[0,0] * psi[1,0].conjugate() + psi[1,0] * psi[0,0].conjugate()
    b_y = -1j * (psi[0,0] * psi[1,0].conjugate() - psi[1,0] * psi[0,0].conjugate())
    b_z = np.abs(psi[1,0])**2 - np.abs(psi[0,0])**2 
    return b_x.real,b_y.real,b_z.real
​
​
## Animated Bloch vector
​
N=2        # number of levels
psi = np.matrix(np.zeros((N,1)))
psi[1,0] = 1
U = np.matrix(np.eye(N))
t = [0]
P_list = [0]
dt = 3e-2
Omega0 = 1.0
w = 10
w0 = 10.0
​
x_list = []
y_list = []
z_list = []
​
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
north_blob, = ax.plot3D([0],[0],[1],'ro')
south_blob = ax.plot3D([0],[0],[-1],'go')
​
theta, phi = np.linspace(0,pi,20), np.linspace(0,2*pi,20)
Theta,Phi = np.meshgrid(theta,phi)
​
x = np.outer(np.cos(phi), np.sin(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.ones(np.size(phi)), np.cos(theta))
   
slider_axes = []
sliders = []
slider_axes.append( plt.axes([0.25, 0.02, 0.65, 0.03]) )
slider_axes.append( plt.axes([0.25, 0.06, 0.65, 0.03]) )
​
sliders.append( Slider(slider_axes[0],'Omega',0,5, valinit=1.0) )
sliders.append( Slider(slider_axes[1],'Delta',-10,10, valinit=0.0) )
​
def update(val):
    global Omega0, w, w0
    Omega0 = sliders[0].val
    w = sliders[1].val + w0
    plt.draw()
    
for s in sliders: s.on_changed(update)
​
def animate(i):
    global psi, Omega0, w, w0
    ti = i*dt
    Omega0 = sliders[0].val
    w = sliders[1].val + w0
    bx,by,bz = bloch_vector(psi)
    psi = lg.expm(-1j*H_int(ti,Omega=Omega0,w = w)*dt) * psi
    
    x_list.append(bx)
    y_list.append(by)
    z_list.append(bz)
​
    ax.clear()
    blob, = ax.plot3D(x_list[-1:],y_list[-1:],z_list[-1:],'bo')
    line, = ax.plot3D(x_list[-30:],y_list[-30:],z_list[-30:],lw=2,ls='-.')
        
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    return (blob,) + (line,) 
​
def init():
    north_blob, = ax.plot3D([0],[0],[1],'ro')
    south_blob, = ax.plot3D([0],[0],[-1],'go')
    #line, = ax.plot3D(np.zeros(30),np.zeros(30),np.zeros(30),lw=2,ls='-.')
    
    plot_args = {'rstride': 1, 'cstride': 1, 'linewidth': 0.05, 'color': 'r', 'alpha': 0.05, 'antialiased':True}
    surf = ax.plot_surface(x, y, z, **plot_args)
    
    # x_axis = ax.plot(np.linspace(0,1.2,30),np.zeros(30),np.zeros(30),'k',lw=0.5)
    # y_axis = ax.plot(np.zeros(30),np.linspace(0,1.2,30),np.zeros(30),'k',lw=0.5)
    # z_axis = ax.plot(np.zeros(30),np.zeros(30),np.linspace(0,1.2,30),'k',lw=0.5)
    
    # Annotate points
    ax.text(0,0,1.2,r"$|0\rangle$")
    ax.text(0,0,-1.2,r"$|1\rangle$")
    # ax.text(1.4,0,0,r"$x$")
    # ax.text(0,1.3,0,r"$y$")
    # 
​
    # ax.elev = 30
    # ax.azim = 25
​
    return (north_blob,) + (south_blob,)
​
ani = animation.FuncAnimation(fig, animate, init_func=init, 
                                interval=1, blit=True)
plt.show()