import numpy as np
import matplotlib.pyplot as plt


t = 10
n = 500
x = np.arange(-t, t, t/n)

def gauss(a, s):
    return np.exp(-(x-a)*(x-a)/(2*s*s))

for i in range(-2, 3, 2):
    for j in range(1, 5, 2):
        plt.plot(x, gauss(i, j), label="a = {}, s = {}".format(i, j))
# Add a legend
plt.legend()

plt.ylabel("f(x)")
plt.xlabel("x")

# Show the plot
plt.show()