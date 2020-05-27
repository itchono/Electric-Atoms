'''
The b's knees
'''
import scipy
import numpy as np
import matplotlib.pyplot as plt

def euler(b0, h, n, t):
    '''
    Use Euler's method to numerically integrate
    '''

    times = np.arange(0, t, t/n)

    b = np.zeros((3, n))

    b[:,0] = b0

    for i in range(1,n):
        b[:,i] = 2*np.cross(h, b[:,i-1]) * t/n + b[:,i-1]

    return b

def graph(b, n, t):
    '''
    Graph results using projection onto 1D axis
    '''
    x = np.arange(0, t, t/n)

    for i in range(3):
        plt.plot(x,b[i,], label="b[{}]".format(i))

    plt.title("Bloch Vector Versus Time for Given Hamiltionian")
    plt.ylabel("Vector Value")
    plt.xlabel("Time (s)")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    h = np.array(eval(input("h vector? input as '[a, b, E]'\n"))) # alpha, 0, E
    b0 = np.array(eval(input("starting b vector? input as '[a, b, c]' (magnitude must be 1)\n"))) # starting at zero point on sphere

    t = 10
    n = 5000
    print("Calculating...")
    b = euler(b0, h, n, t)
    print("Graphing...")
    graph(b, n, t)

    


