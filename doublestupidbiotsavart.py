import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# store shape as a series of points

# coil points stored as [[x, y, z], [x, y, z]] in cm
# current stored in amps

# assume infinitely thin conductor

COIL = np.array([[0,0,0], [10, 0, 0], [10, 10, 0], [20, 10, 0]]).T
print(COIL)
CURRENT = 1

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces
    '''

    newcoil = np.zeros(3)

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
            np.r_[newcoil, (start + diff * j/stepnumber)]

    return newcoil

def calculateField(coil, current, position):
    '''
    Calculates magnetic field vector as a result of some position vector tuple (x, y, z)
    '''

    FACTOR = 10**(-7) # equals mu_0 / 4pi

    B = np.zeros(3)

    for i in range(len(coil)-1):

        start = coil[i]
        end = coil[i+1]
        # determine start and end points of our line segment

        dl = end - start
        midstep = (start + end)/2 
        # this is the effective position of our element (r' in the paper)

        db = current * np.cross(dl, (position - midstep)) * FACTOR / (np.linalg.norm(position - midstep) ** 3) 
        # Biot-Savart Law

        B += db
    
    return B

            


if __name__ == "__main__":
    chopped = sliceCoil(COIL, 1)
    print("Slices generated.")

    print(chopped)

    '''B = calculateField(chopped, CURRENT, (0.05, 0, 0.01))
    print(B)'''



