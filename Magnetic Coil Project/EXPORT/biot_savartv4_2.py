import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker

'''
B-field calculator using Biot-Savart law 

You will need to provide:
- series of points in (x, y, z, I) describing the geometry of a coil in a text file.
    - Each section of coil will have some amount of current flowing through it, in the forwards direction of the points in the coil.
    - The current in a given segment of coil is defined by the current listed at the starting point of the coil
    - Ex. P1------P2 is on our coil. The current P1 --> P2 is given by I1 (assuming P1 = (x1, y1, z1, I1))

- rectangular volume over which the fields need to be calculated
- resolution at which fields should be calculated
- all lengths in cm, B-field in G

version history:
    v2: Mingde Yin  May 31, 2020. Accelerated using numpy meshgrids
    v2.1: Jun 1, 2020. tkinter dialogs for opening & saving files. Defaults of 1 cm resolution in calculation.
    v3: Ryan Zazo  Jun 1, 2020. Plotting code integrated.
    v3.1: Minor cosmetic improvements to plot.
    v3.2: 3D plot of coil geometry.
    v3.3: Plotted B-fields together but code is long.
    v3.5: all B-field plots together
    v3.7: B-fields plotted together with 50 levels (now works on windows) and combined v3.3 and v3.5
    v3.8: Changed up all np.aranges to np.linspaces and changed up the plotting code to work with non-integer step sizes and non-integer levels
    v4: Mingde Yin June 9, 2020 Using Richardson Extrapolation for midpoint rule to improve accuracy (5 to 30x better at 1.4x speed penalty), tweaked linspaces to correctly do step size
    v4.1: Minor change in function indexing to use more numpy, cleaning up for export
    v4.2: Changed the linspaces a bit to make everything more symmetric

wishlist:
    1. improve plot_coil with different colors for different values of current?
    2. improve plot_fields to reduce space between Bz plot and colorbar ---- done
'''

def parseCoil(filename):
    '''
    Parses 4 column CSV into x,y,z,I slices for coil
    '''
    with open(filename, "r") as f: return np.array([[eval(i) for i in line.split(",")] for line in f.read().splitlines()]).T

'''
FILE FORMAT for coil.txt:

x1,y1,z1,I1
x2,y2,z2,I2
.
.
xn,yn,zn,In
'''

def sliceCoil(coil, steplength):
    '''
    Slices a coil into smaller steplength-sized pieces based on the coil resolution
    '''
    def interpolatePoints(p1, p2, parts):
        '''
        Produces a series of linearly spaced points between two given points in R3+I (retains same current)
        '''
        return np.column_stack((np.linspace(p1[0], p2[0], parts+1), np.linspace(p1[1], p2[1], parts+1),
                np.linspace(p1[2], p2[2], parts+1), p1[3] * np.ones((parts+1))))

    newcoil = np.zeros((1, 4)) # fill with dummy first column

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

    ## Force the coil to have an even number of segments, for Richardson Extrapolation
    if newcoil.shape[0] %2 != 0: newcoil = np.vstack((newcoil, newcoil[-1,:]))
    
    return newcoil[1:,:].T # return non-dummy columns

def calculateField(coil, x, y, z):
    '''
    Calculates magnetic field vector as a result of some position and current x, y, z, I
    [In the same coordinate system as the coil]

    Coil: Input Coil Positions, already sub-divided into small pieces using sliceCoil
    x, y, z: position in cm
    
    Output B-field is a 3-D vector in units of G
    '''
    FACTOR = 0.1 # = mu_0 / 4pi when lengths are in cm, and B-field is in G

    def BSintegrate(start, end):
        '''
        Produces tiny segment of magnetic field vector (dB) using the midpoint approximation over some interval

        for future optimization: Get this to work with meshgrids
        '''
        dl = (end-start).T
        mid = (start+end)/2
        position = np.array((x-mid[0], y-mid[1], z-mid[2])).T
        mag = np.sqrt((x-mid[0])**2 + (y-mid[1])**2 + (z-mid[2])**2)

        return start[3] * np.cross(dl[:3], position) / np.array((mag ** 3, mag ** 3, mag ** 3)).T

        # Biot-Savart Law
        # current flowing in this segment is represented by start[3]

    B = 0

    # midpoint integration with 1 layer of Richardson Extrapolation
    starts, mids, ends = coil[:,:-1:2], coil[:,1::2], coil[:,2::2]

    for start, mid, end in np.nditer([starts, mids, ends], flags=['external_loop'], order='F'):
        # use numpy fast indexing
        fullpart = BSintegrate(start, end) # stage 1 richardson
        halfpart = BSintegrate(start, mid) + BSintegrate(mid, end) # stage 2 richardson

        B += 4/3 * halfpart - 1/3 * fullpart # richardson extrapolated midpoint rule
    
    return B * FACTOR # return SUM of all components as 3 (x,y,z) meshgrids for (Bx, By, Bz) component when evaluated using produceTargetVolume

def produceTargetVolume(coil, box_size, startpoint, vol_resolution):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.
​
    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    BoxSize: (x, y, z) dimensions of the box in cm
    Startpoint: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    Steplength: Spatial resolution (in cm)
    '''
    x = np.linspace(startpoint[0], box_size[0],int(box_size[0]/vol_resolution)+1)
    y = np.linspace(startpoint[1], box_size[1],int(box_size[1]/vol_resolution)+1)
    z = np.linspace(startpoint[2], box_size[2],int(box_size[2]/vol_resolution)+1)
    # Generate points at regular spacing, incl. end points
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # NOTE: Requires axes to be flipped in order for meshgrid to have the correct dimensional order
    # it's just a weird thing with numpy

    return calculateField(coil, X,Y,Z)

def getFieldVector(targetVolume, position, startpoint, volumeresolution):
    '''
    Returns the B vector [Bx, By, Bz] components in a generated Target Volume at a given position tuple (x, y, z) in a coordinate system

    Startpoint: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    VolumeResolution: Division of volumetric meshgrid (generate a point every VolumeResolution cm)
    '''
    relativePosition = ((np.array(position) - np.array(startpoint)) / volumeresolution).astype(int)
    # adjust to the meshgrid's system

    if (relativePosition < 0).any(): return ("ERROR: Out of bounds! (negative indices)")

    try: return targetVolume[relativePosition[0], relativePosition[1], relativePosition[2], :]
    except: return ("ERROR: Out of bounds!")
    # basic error checking to see if you actually got a correct input/output

'''
EXTERNAL USAGE

Import all functions in this file

- The coil will load in from the selected file
- You must then slice the coil
- You must then produce a Target Volume using the coil
- Then, you can either use getFieldVector(), or index on your own

- If you are indexing on your own, remember to account for the offset (starting point), and spatial resolution
- something like <relativePosition = ((np.array(position) - np.array(startpoint)) / volumeresolution).astype(int)>
'''

def writeTargetVolume(input_filename,output_filename, boxsize, startpoint, 
                        coilresolution=1, volumeresolution=1):
    '''
    Takes a coil specified in input_filename, generates a target volume, and saves the generated target volume to output_filename.

    BoxSize: (x, y, z) dimensions of the box in cm
    Startpoint: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    CoilResolution: How long each coil subsegment should be
    VolumeResolution: Division of volumetric meshgrid (generate a point every VolumeResolution cm)
    '''
    coil = parseCoil(input_filename) 
    chopped = sliceCoil(coil, coilresolution)
    targetVolume = produceTargetVolume(chopped, boxsize, startpoint, volumeresolution)

    with open(output_filename, "wb") as f:
        np.save(f, targetVolume)


def readTargetVolume(filename):
    '''
    Takes the name of a saved target volume and loads the B vector meshgrid.
    Returns None if not found.
    '''
    targetVolume = None

    try:
        with open(filename, "rb") as f:
            targetVolume = np.load(f)
        return targetVolume
    except:
        pass

## plotting routines

def plot_fields(Bfields,startpoint,box_size,vol_resolution,which_plane='z',level=0,num_contours=50):
    # filled contour plot of Bx, By, and Bz on a chosen slice plane

    # M.Y: Changed linspace to generate the correct amount of points
    X = np.linspace(startpoint[0], box_size[0],int(box_size[0]/vol_resolution)+1)
    Y = np.linspace(startpoint[1], box_size[1],int(box_size[1]/vol_resolution)+1)
    Z = np.linspace(startpoint[2], box_size[2],int(box_size[2]/vol_resolution)+1)

    if which_plane=='x':

        converted_level = np.where(X >= level)
        B_sliced = [Bfields[converted_level[0][0],:,:,i].T for i in range(3)]
        x_label,y_label = "y","z"
        x_array,y_array = Y,Z
    elif which_plane=='y':
        converted_level = np.where(Y >= level)
        B_sliced = [Bfields[:,converted_level[0][0],:,i].T for i in range(3)]
        x_label,y_label = "x","z"
        x_array,y_array = X,Z
    else:
        converted_level = np.where(Z >= level)
        B_sliced = [Bfields[:,:,converted_level[0][0],i].T for i in range(3)]
        x_label,y_label = "x","y"
        x_array,y_array = X,Y
    
    Bmin,Bmax = np.amin(B_sliced),np.amax(B_sliced)
    
    component_labels = ['x','y','z']
    fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(10,5))
    axes[0].set_ylabel(y_label + " (cm)")

    for i in range(3):
        contours = axes[i].contourf(x_array,y_array,B_sliced[i],
                                    vmin=Bmin,vmax=Bmax,
                                    cmap=cm.magma,levels=num_contours)
        axes[i].set_xlabel(x_label + " (cm)")
        axes[i].set_title("$\mathcal{B}$"+"$_{}$".format(component_labels[i]))
    
    axes[3].set_aspect(20)
    fig.colorbar(contours,cax=axes[3],extend='both')
    
    plt.tight_layout()


    plt.show()
    
def plot_coil(input_filename):
    coil_points = parseCoil(input_filename)
    fig = plt.figure()
    tick_spacing = 2
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_zlabel("$z$ (cm)")
    ax.plot3D(coil_points[0,:],coil_points[1,:],coil_points[2,:],lw=2)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.show()