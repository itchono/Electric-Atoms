import numpy as np
import pickle
import time
from numba import njit

import cProfile

'''
BIOT-SAVART Calculator
Accelerated using numpy meshgrids

You will need to provide:
- A series of points in (x, y, z) describing the geometry of a coil, used as our magnet. [coil.txt]
- A rectangular box of a certain size, with a certain offset position
- An amount of current going into the coil

- Resolution at which you want to conduct your measurements

Please see configuration section.

Mingde Yin
May 29, 2020
'''

'''
CONFIGURATION OF VARIABLES
'''
CURRENT = 1 # Ampere
BOX_SIZE = (30, 15, 15) # dimensions of box in cm (x, y, z)
BOX_OFFSET = (-5, -2.5, -7.5) # where the bottom left corner of the box is w/r to the coil coordinate system.

COIL_RESOLUTION = 1 # cm; affects runtime of calculation process linearly, and increases precision up to a point
VOLUME_RESOLUTION = 1 # cm; affects runtime of calculation process in n^3, and size of resulting model


def parseCoil(filename):
    with open(filename, "r") as f:
        lines = [eval(l) for l in f.read().splitlines()] # tuples
        return np.array(lines).T

COIL = parseCoil("coil.txt")
# cartesian points in cm
'''
FILE FORMAT
Tuples Stored in form:
(x1, y1, z1)
(x2, y2, z2)
.
.
.
(xn, yn, zn)
'''

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces based on the coil resolution
    '''
    
    def getEquidistantPoints(p1, p2, parts):
        '''
        Produces a series of linearly spaced points between two given points in R3
        '''
        return np.column_stack((np.linspace(p1[0], p2[0], parts+1),
                np.linspace(p1[1], p2[1], parts+1),
                np.linspace(p1[2], p2[2], parts+1)))

    newcoil = np.zeros((1, 3)) # fill with dummy column

    segment_starts = coil[:,:-1]
    segment_ends = coil[:,1:]
    # determine start and end of each segment

    segments = segment_ends-segment_starts
    segment_lengths = np.apply_along_axis(np.linalg.norm, 0, segments)
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

def calculateField(coil, current, x, y, z):
    '''
    Calculates magnetic field vector as a result of some position x, y, z
    [In the same coordinate system as the coil]

    Coil: Input Coil Positions, already sub-divided into small pieces using sliceCoil
    '''

    FACTOR = 10**(-7) # equals mu_0 / 4pi

    B = 0

    for i in range(coil.shape[1]-1):
        start = coil[:,i]
        end = coil[:,i+1]
        # determine start and end points of our line segment

        dl = end - start
        dl = dl.T
        midstep = (start + end)/2 
        # this is the effective position of our element (r' in the paper)
        # midpoint approximation for numerical integration

        difference = np.array([x-midstep[0], y-midstep[1], z-midstep[2]]).T

        mag = np.sqrt((x-midstep[0])**2 + (y-midstep[1])**2 + (z-midstep[2])**2)

        db = current * np.cross(dl, difference) * FACTOR / np.array((mag ** 3, mag ** 3, mag ** 3)).T 
        # Biot-Savart Law

        B += db
    
    return B # return SUM of all components

def produceModel(coil, current, startpoint, steplength):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.

    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    Current: Amount of current in amps flowing through coil from [start of coil] to [end of coil]
    Startpoint: (x, y, z) = (0, 0, 0) position of the box (30 x 15 x 15) cm
    Steplength: Spatial resolution (in cm)

    Optimized using numpy meshgrid
    '''

    x = np.arange(startpoint[0], startpoint[0] + BOX_SIZE[0] + steplength, steplength)
    y = np.arange(startpoint[1], startpoint[1] + BOX_SIZE[1] + steplength, steplength)
    z = np.arange(startpoint[2], startpoint[2] + BOX_SIZE[2] + steplength, steplength)
    # this model includes endpoints

    X,Y,Z = np.meshgrid(x, y, z, indexing='ij')

    return calculateField(coil, current, X,Y,Z)

def getFieldVector(model, position):
    '''
    Returns the B vector [Bx, By, Bz] components in a generated model at a given position tuple (x, y, z) in a coordinate system
    '''

    relativePosition = ((np.array(position) - np.array(BOX_OFFSET)) / VOLUME_RESOLUTION).astype(int)
    # adjust to the meshgrid's system
    # print("Access indices: {}".format(relativePosition)) # --> if you need to debug the mesh grid

    try: return model[relativePosition[2], relativePosition[1], relativePosition[0], :]
    except: return ("ERROR: Out of bounds!")
    # basic error checking to see if you actually got a correct input/output


if __name__ == "__main__":
    print("Generating model...")
    t_start = time.perf_counter()
    chopped = sliceCoil(COIL, COIL_RESOLUTION)  
    model = produceModel(chopped, CURRENT, BOX_OFFSET, VOLUME_RESOLUTION)
    t_end = time.perf_counter()

    print("Model made in {:.4f}s, of shape {}".format(t_end-t_start, model.shape))

    print("Starting position: {} cm with stepsize of {} cm".format(BOX_OFFSET, VOLUME_RESOLUTION))

    print("Please input the position at which you want to see the B vector...")
    
    position = (eval(input("x?\n")), eval(input("y?\n")), eval(input("z?\n")))
    
    print(getFieldVector(model, position), "Gs at {} cm".format(position))


'''
EXTERNAL USAGE

Import all functions in this file

- The coil will load in automatically from the file "coil.txt"
- You must then slice the coil
- You must then produce a model using the coil
- Then, you can either use tgetFieldVector(), or index on your own
- NOTE: The model is stored in form [Z, Y, X, b-component] (coordinates are reversed)
'''
    
    

    



