import numpy as np
import time

'''
BIOT-SAVART Calculator v2
Accelerated using numpy meshgrids

You will need to provide:
- A series of points in (x, y, z, I) describing the geometry of a coil, used as our magnet. [coil.txt]
    - Each section of coil will have some amount of current flowing through it, in the forwards direction of the points in the coil.
    - The current in a given segment of coil is defined by the current listed at the starting point of the coil
    - Ex. P1------P2 is on our coil. The current P1 --> P2 is given by I1 (assuming P1 = (x1, y1, z1, I1))

- A rectangular box of a certain size, with a certain offset position
- Resolution at which you want to conduct your measurements

Please see configuration section.

Mingde Yin
May 29, 2020
'''

'''
CONFIGURATION OF VARIABLES
'''

BOX_SIZE = (30, 15, 15) # dimensions of box in cm (x, y, z)
BOX_OFFSET = (-5, -2.5, -7.5) # where the bottom left corner of the box is w/r to the coil coordinate system.

COIL_RESOLUTION = 1 # cm; affects runtime of calculation process linearly, and increases precision up to a point
VOLUME_RESOLUTION = 1 # cm; affects runtime of calculation process in n^3, and size of resulting Target Volume


def parseCoil(filename):
    '''
    Parses 4 column CSV into x,y,z,I slices for coil
    '''
    with open(filename, "r") as f:
        return np.array([[eval(i) for i in line.split(",")] for line in f.read().splitlines()]).T

COIL = parseCoil("coil.txt")
# cartesian + current points in cm, Amperes
'''
FILE FORMAT
CSV Stored in form:
x1,y1,z1,I1
x2,y2,z2,I2
.
.
.
xn,yn,zn,In
'''

# TODO: Option to do points and current as separate files?

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces based on the coil resolution
    '''
    def interpolatePoints(p1, p2, parts):
        '''
        Produces a series of linearly spaced points between two given points in R3+I (retains same current)
        '''
        return np.column_stack((np.linspace(p1[0], p2[0], parts+1),
                np.linspace(p1[1], p2[1], parts+1),
                np.linspace(p1[2], p2[2], parts+1), p1[3] * np.ones((parts+1))))

    newcoil = np.zeros((1, 4)) # fill with dummy column

    segment_starts = coil[:,:-1]
    segment_ends = coil[:,1:]
    # determine start and end of each segment

    segments = segment_ends-segment_starts
    segment_lengths = np.apply_along_axis(np.linalg.norm, 0, segments)
    # create segments; determine start and end of each segment, as well as segment lengths

    # chop up into smaller bits (elements)
    stepnumbers = (segment_lengths/steplength).astype(int)
    # determine how many steps we must chop each segment into

    for i in range(segments.shape[1]):
        newrows = interpolatePoints(segment_starts[:,i], segment_ends[:,i], stepnumbers[i])
        # set of new interpolated points to feed in
        newcoil = np.vstack((newcoil, newrows))

    return newcoil[1:,:].T # return non-dummy columns

def calculateField(coil, x, y, z):
    '''
    Calculates magnetic field vector as a result of some position and current x, y, z, I
    [In the same coordinate system as the coil]

    Coil: Input Coil Positions, already sub-divided into small pieces using sliceCoil
    
    Output B-field is a 3-D vector in units of G
    '''

    FACTOR = 0.1 # equals mu_0 / 4pi for when all lengths are in cm, used to return B field in G.

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

        db = start[3] * np.cross(dl[:3], difference) * FACTOR / np.array((mag ** 3, mag ** 3, mag ** 3)).T
        # Biot-Savart Law
        # current equals start[3]

        # Needs the whole transpose thing because "operands could not be broadcast together with shapes (31,16,17,3) (17,16,31)" otherwise
        # quirk with meshgrids
        B += db
    
    return B # return SUM of all components
    # evaluated using produceTargetVolume

def produceTargetVolume(coil, startpoint, steplength):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.

    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    Startpoint: (x, y, z) = (0, 0, 0) position of the box (30 x 15 x 15) cm
    Steplength: Spatial resolution (in cm)
    '''
    x = np.arange(startpoint[0], startpoint[0] + BOX_SIZE[0] + steplength, steplength)
    y = np.arange(startpoint[1], startpoint[1] + BOX_SIZE[1] + steplength, steplength)
    z = np.arange(startpoint[2], startpoint[2] + BOX_SIZE[2] + steplength, steplength)
    # Generate points at regular spacing, incl. end points
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # NOTE: Requires axes to be flipped in order for meshgrid to work as intended
    # it's just a weird thing with numpy

    return calculateField(coil, X,Y,Z)

def getFieldVector(targetVolume, position):
    '''
    Returns the B vector [Bx, By, Bz] components in a generated Target Volume at a given position tuple (x, y, z) in a coordinate system
    '''

    relativePosition = ((np.array(position) - np.array(BOX_OFFSET)) / VOLUME_RESOLUTION).astype(int)
    # adjust to the meshgrid's system
    
    # print("Access indices: {}".format(relativePosition)) # --> if you need to debug the mesh grid

    if (relativePosition < 0).any(): return ("ERROR: Out of bounds! (negative indices)")

    try: return targetVolume[relativePosition[0], relativePosition[1], relativePosition[2], :]
    except: return ("ERROR: Out of bounds!")
    # basic error checking to see if you actually got a correct input/output


'''
EXTERNAL USAGE

Import all functions in this file

- The coil will load in automatically from the file "coil.txt"
- You must then slice the coil
- You must then produce a Target Volume using the coil
- Then, you can either use getFieldVector(), or index on your own
'''

def writeField(filename):
    '''
    takes a coil specified in coil.txt and write out the B-field in the target volume into 3 separate text files (Bx.txt, By.txt, Bz.txt). 
    '''
    coil = parseCoil(filename)

    chopped = sliceCoil(coil, COIL_RESOLUTION)

    targetVolume = produceTargetVolume(chopped, (-5, -2.5, -7.5), VOLUME_RESOLUTION)

    np.savetxt("Bx.txt", targetVolume[:,:,:,0])
    np.savetxt("By.txt", targetVolume[:,:,:,1])
    np.savetxt("Bz.txt", targetVolume[:,:,:,2])

def readField(BxName, ByName, BzName):
    '''
    Takes stores Bx, By, Bz, and reloads into memory
    '''

if __name__ == "__main__":
    print("Generating Points...")
    t_start = time.perf_counter()
    chopped = sliceCoil(COIL, COIL_RESOLUTION)
    targetVolume = produceTargetVolume(chopped, BOX_OFFSET, VOLUME_RESOLUTION)
    t_end = time.perf_counter()

    print("Target Volume made in {:.4f}s, of shape {}".format(t_end-t_start, targetVolume.shape))
    print("Starting position: {} cm with stepsize of {} cm".format(BOX_OFFSET, VOLUME_RESOLUTION))
    
    try:
        while True:
            print("Please input the position at which you want to see the B vector...")
        
            position = (eval(input("x?\t")), eval(input("y?\t")), eval(input("z?\t")))
            
            print(getFieldVector(targetVolume, position), "Gs at {} cm".format(position))
    except KeyboardInterrupt:
        print("DONE")

    writeField("coil.txt")    


