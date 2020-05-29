import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from numba import njit

import cProfile


# store shape as a series of points

# coil points stored as columns of 3 x n matrix, in cm
# current stored in amps

# assume infinitely thin conductor

'''
CFG
'''
COIL = np.array([[0,0,0], [10, 0, 0], [10, 10, 0], [20, 10, 0]]).T
# cartesian points in cm

CURRENT = 1 # Ampere
BOX_SIZE = (30, 15, 15) # dimensions of box in cm (x, y, z)
BOX_OFFSET = (-7.5, -2.5, -2.5) # where the bottom left corner of the box is w/r to the coil coordinate system.

COIL_RESOLUTION = 1 # cm
VOLUME_RESOLUTION = 1 # cm

def getEquidistantPoints(p1, p2, parts):
    # from stackoverflow
    return np.column_stack((np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1),
               np.linspace(p1[2], p2[2], parts+1)))

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces
    Takes on the order of 1-3 ms currently for the simple coil
    '''
    newcoil = np.zeros((1, 3)) # fill with dummy column

    segment_starts = coil[:,:-1]
    segment_ends = coil[:,1:]
    # determine start and end of each segment

    segments = segment_ends-segment_starts
    segment_lengths = mag_r = np.apply_along_axis(np.linalg.norm, 0, segments)
    # create segments; determine start and end of each segment, as well as segment lengths

    # chop up into smaller bits (elements)
    stepnumbers = (segment_lengths/steplength).astype(int)
    # determine how many steps we must chop each segment into

    for i in range(segments.shape[0]):
        # still slow; TODO how to turn into numpy?
        newrows = getEquidistantPoints(segment_starts[:,i], segment_ends[:,i], stepnumbers[i])
        # set of new interpolated points to feed in
        newcoil = np.vstack((newcoil, newrows))

    return newcoil[1:,:].T # return non-dummy columns

@njit
def calculateField(coil, current, position):
    '''
    Calculates magnetic field vector as a result of some position vector tuple (x, y, z)
    attempting to use numba for performance improvement
    '''
    position = np.array(position)
    position = np.reshape(position, (1,3))

    FACTOR = 10**(-7) # equals mu_0 / 4pi

    B = np.zeros((1,3))

    for i in range(coil.shape[1]-1):
        start = coil[:,i]
        end = coil[:,i+1]
        # determine start and end points of our line segment

        dl = end - start
        dl = dl.T
        midstep = (start + end)/2 
        midstep = midstep.T
        # this is the effective position of our element (r' in the paper)

        # WEIRD REALIGNMENTS FOR NUMBA TO WORK PLEASE

        db = current * np.cross(dl, (position - midstep)) * FACTOR / (np.linalg.norm(position - midstep) ** 3) 
        # Biot-Savart Law

        B += db
    
    return B[0]

@njit
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

def profiler():
    chopped = sliceCoil(COIL, COIL_RESOLUTION)  
    model = produceModel(chopped, CURRENT, BOX_OFFSET, VOLUME_RESOLUTION)
    # after 12 ish seconds of init, subsequent calls take 0.5 seconds
    # 619 function calls (559 primitive calls) in 0.501 seconds

if __name__ == "__main__":

    profiler()

    pr = cProfile.Profile()
    pr.enable()

    profiler()

    pr.disable()

    pr.print_stats()
    
    

    



