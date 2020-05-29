import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time


# store shape as a series of points

# coil points stored as columns of 3 x n matrix, in cm
# current stored in amps

# assume infinitely thin conductor

COIL = np.array([[0,0,0], [10, 0, 0], [10, 10, 0], [20, 10, 0]]).T
CURRENT = 1

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces
    '''

    newcoil = np.zeros((3, 1)) # fill with dummy column

    for i in range(coil.shape[1]-1):

        start = coil[:,i]
        end = coil[:,i+1]
        # determine start and end points of our line segment

        diff = end - start
        # determine the line segment vector

        dist = np.linalg.norm(diff)
        # determine how big this gap is

        # chop up into smaller bits (elements)
        stepnumber = int(dist/steplength)

        for j in range(stepnumber):
            newcol = (start + diff * j/stepnumber)
            # change j to a vector? --> start + diff * j directly without for loop
            newcoil = np.column_stack((newcoil, newcol))

    return newcoil[:,1:] # return non-dummy columns

def calculateField(coil, current, x, y, z, component):
    '''
    Calculates magnetic field vector as a result of some position vector in numpy
    '''

    FACTOR = 10**(-7) # equals mu_0 / 4pi

    B = 0

    for i in range(coil.shape[1]-1):
        start = coil[:,i]
        end = coil[:,i+1]
        # determine start and end points of our line segment

        dl = end - start
        midstep = (start + end)/2 
        # this is the effective position of our element (r' in the paper)

        positionTerm = (x-midstep[0], y - midstep[1], z -midstep[2]) # (r - r')

        db = current * np.cross(dl, positionTerm) * FACTOR / (np.linalg.norm(positionTerm) ** 3) 
        # Biot-Savart Law

        B += db[component]
    
    return B

def produceModel(coil, current, startpoint, steplength):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the space

    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    Current: Amount of current in amps flowing through coil from [start of coil] to [end of coil]
    Startpoint: (x, y, z) = (0, 0, 0) position of the box (30 x 15 x 15) cm
    Steplength: Spatial resolution (in cm)
    '''

    BOX_SIZE = (5, 2, 2) # dimensions of box

    x = np.arange(0, BOX_SIZE[0], steplength)
    y = np.arange(0, BOX_SIZE[1], steplength)
    z = np.arange(0, BOX_SIZE[2], steplength)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    modelX = calculateField(coil, current, X, Y, Z, 0)
    modelY = calculateField(coil, current, X, Y, Z, 1)
    modelZ = calculateField(coil, current, X, Y, Z, 2)

    return (modelX, modelY, modelZ)

if __name__ == "__main__":
    chopped = sliceCoil(COIL, 1)
    print("Slices generated in {}s".format(time.perf_counter()))
    t_start = time.perf_counter()

    x = chopped[0,:]
    y = chopped[1,:]
    z = chopped[2,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    print("Generating model...")
    model = produceModel(chopped, CURRENT, (-5, -2.5, -7.5), 1)

    print("Model made in {}s".format(time.perf_counter()-t_start))

    '''with open("model.md", "wb") as f:
        pickle.dump(model, f)
        print("Model saved.")'''

    '''boxpoints = np.array(list(model.keys()))

    x = boxpoints[:,0]
    y = boxpoints[:,1]
    z = boxpoints[:,2]

    ax.scatter(x, y, z, c="#010101a0")

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')

    plt.show()'''



'''
Meshgrid:
Store 3 meshgrids, x, y, z --> uniform spacing

<will need to do a smart spatial conversion to actual box conversion>
'''