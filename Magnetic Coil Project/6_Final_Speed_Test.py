import numpy as np
import pickle
import time
from numba import njit

import cProfile

'''
BIOT-SAVART Calculator

Given: Coil as a series of points in np array, Rectangular Search Box, Precision, Current into Coil

Mingde Yin
May 28, 2020
'''

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


def calculateField(coil, current, x, y, z):
    '''
    Calculates magnetic field vector as a result of some position x, y, z
    [In the same coordinate system as the coil]
    '''
    position = np.array([x,y,z])

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

        # WEIRD REALIGNMENTS FOR NUMBA TO WORK PLEASE

        difference = np.array([x-midstep[0], y-midstep[1], z-midstep[2]]).T

        mag = np.sqrt((x-midstep[0])**2 + (y-midstep[1])**2 + (z-midstep[2])**2)

        db = current * np.cross(dl, difference) * FACTOR / np.array((mag ** 3, mag ** 3, mag ** 3)).T 
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

    Optimized using numpy meshgrid
    '''

    x = np.arange(startpoint[0], startpoint[0] + BOX_SIZE[0], steplength)
    y = np.arange(startpoint[1], startpoint[1] + BOX_SIZE[1], steplength)
    z = np.arange(startpoint[2], startpoint[2] + BOX_SIZE[2], steplength)

    X,Y,Z = np.meshgrid(x, y, z, indexing='ij')

    return calculateField(coil, current, X,Y,Z)

def profiler():
    chopped = sliceCoil(COIL, COIL_RESOLUTION)  
    model = produceModel(chopped, CURRENT, BOX_OFFSET, VOLUME_RESOLUTION)
    print(model.shape)
    # 0.045 seconds approx.
    # 3254 function calls (3109 primitive calls) in 0.043 seconds
    # almost 1000x faster, 6000 x fewer function calls

def profiler():

    chopped = sliceCoil(COIL, COIL_RESOLUTION)  
    model = produceModel(chopped, CURRENT, BOX_OFFSET, VOLUME_RESOLUTION)
    print(model.shape)

def speedtest():
    chopped = sliceCoil(COIL, 1)

    times = []

    for i in range(100):
        t_start = time.perf_counter()

        model = produceModel(chopped, CURRENT, (-7.5, -2.5, i), 1)

        t_end = time.perf_counter()

        print("T: {}".format(t_end-t_start))

        times.append(t_end-t_start)

    print("Average: {}".format(sum(times)/len(times)))

if __name__ == "__main__":
    cProfile.run("profiler()")

    speedtest()
    
    

    



