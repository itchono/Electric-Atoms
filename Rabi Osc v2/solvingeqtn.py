import numpy as np
from scipy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt, e
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


T = 20
DELTA = 6
NUM_PI = 2

OMEGA = pi/T

def equation_system(t, v, omega, delta):
    ce, cg = v

    cedot = -1j * omega/2 * e**(1j*delta*t) * cg

    cgdot = -1j * omega/2 * e**(1j*delta*t) * ce

    return cedot, cgdot


def regularplot():
    t = np.linspace()

if __name__ == "__main__":
    # init
    t = np.linspace(0,NUM_PI*T,50)
    r_init = np.array([0, 1]).astype(complex)

    # solve
    solution = solve_ivp(equation_system, [0,NUM_PI*T], r_init,t_eval=t, args=(OMEGA, DELTA))

    t = solution.t
    ce, cg = solution.y

    ce2 = np.abs(ce)**2
    cg2 = np.abs(cg)**2

    # plot
    plt.plot(t, cg2, lw=2,label=r"$|c_g|^{2}$")
    plt.plot(t, ce2, lw=2,label=r"$|c_e|^{2}$")
    plt.title('Rabi Oscillations')
    plt.legend()
    plt.show()

