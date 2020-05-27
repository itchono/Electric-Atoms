import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


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
            newcoil = np.column_stack((newcoil, newcol))

    return newcoil[:,1:] # return non-dummy columns

def calculateField(coil, current, position):
    '''
    Calculates magnetic field vector as a result of some position vector tuple (x, y, z)
    '''

    FACTOR = 10**(-7) # equals mu_0 / 4pi

    B = np.zeros(3)

    for i in range(coil.shape[1]-1):
        start = coil[:,i]
        end = coil[:,i+1]
        # determine start and end points of our line segment

        dl = end - start
        midstep = (start + end)/2 
        # this is the effective position of our element (r' in the paper)

        db = current * np.cross(dl, (position - midstep)) * FACTOR / (np.linalg.norm(position - midstep) ** 3) 
        # Biot-Savart Law

        B += db
    
    return B

def produceModel(coil, current, startpoint, steplength):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the space

    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    Current: Amount of current in amps flowing through coil from [start of coil] to [end of coil]
    Startpoint: (x, y, z) = (0, 0, 0) position of the box (30 x 15 x 15) cm
    Steplength: Spatial resolution (in cm)
    '''

    model = {}

    BOX_SIZE = (30, 15, 15) # dimensions of box

    for x in range(0, BOX_SIZE[0] + steplength, steplength):
        for y in range(0, BOX_SIZE[1] + steplength, steplength):
            for z in range(0, BOX_SIZE[2] + steplength, steplength):
                # print("Point {}".format((x,y,z)))
                model[(x+startpoint[0],y+startpoint[1],z+startpoint[2])] = calculateField(coil, current, (x+startpoint[0],y+startpoint[1],z+startpoint[2]))

    return model

if __name__ == "__main__":
    chopped = sliceCoil(COIL, 1)
    print("Slices generated.")

    x = chopped[0,:]
    y = chopped[1,:]
    z = chopped[2,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    model = produceModel(chopped, CURRENT, (-5, -2.5, -7.5), 2)

    with open("model.md", "wb") as f:
        pickle.dump(model, f)
        print("Model saved.")

    boxpoints = np.array(list(model.keys()))

    x = boxpoints[:,0]
    y = boxpoints[:,1]
    z = boxpoints[:,2]

    ax.scatter(x, y, z, c="#010101a0")

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')

    plt.show()



