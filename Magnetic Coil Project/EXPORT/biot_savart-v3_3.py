import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker

'''
BIOT-SAVART Calculator

You will need to provide:
- A series of points in (x, y, z, I) describing the geometry of a coil, used as our magnet. [coil.txt]
    - Each section of coil will have some amount of current flowing through it, in the forwards direction of the points in the coil.
    - The current in a given segment of coil is defined by the current listed at the starting point of the coil
    - Ex. P1------P2 is on our coil. The current P1 --> P2 is given by I1 (assuming P1 = (x1, y1, z1, I1))

- A rectangular box of a certain size, with a certain offset position
- Resolution at which you want to conduct your measurements

version history:
    v2: Mingde Yin  May 31, 2020. Accelerated using numpy meshgrids
    v2.1: Jun 1, 2020. tkinter dialogs for opening & saving files. Defaults of 1 cm resolution in calculation.
    v3: Ryan Zazo  Jun 1, 2020. Plotting code integrated.
    v3.1: Minor cosmetic improvements to plot.
    v3.2: 3D plot of coil geometry.
'''

def parseCoil(filename):
    '''
    Parses 4 column CSV into x,y,z,I slices for coil
    '''
    with open(filename, "r") as f:
        return np.array([[eval(i) for i in line.split(",")] for line in f.read().splitlines()]).T

'''
FILE FORMAT for coil.txt
CSV Stored in form:
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
        return np.column_stack((np.linspace(p1[0], p2[0], parts+1),
                np.linspace(p1[1], p2[1], parts+1),
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

    return newcoil[1:,:].T # return non-dummy columns

def calculateField(coil, x, y, z):
    '''
    Calculates magnetic field vector as a result of some position and current x, y, z, I
    [In the same coordinate system as the coil]

    Coil: Input Coil Positions, already sub-divided into small pieces using sliceCoil
    x, y, z: position in cm
    
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
        # current in this case is represented by start[3]

        # Needs the whole transpose thing because "operands could not be broadcast together with shapes (31,16,17,3) (17,16,31)" otherwise
        # quirk with meshgrids
        B += db
    
    return B # return SUM of all components as 3 (x,y,z) meshgrids for (Bx, By, Bz) component when evaluated using produceTargetVolume

def produceTargetVolume(coil, boxsize, startpoint, steplength):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.

    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    BoxSize: (x, y, z) dimensions of the box in cm
    Startpoint: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    Steplength: Spatial resolution (in cm)
    '''
    x = np.arange(startpoint[0], startpoint[0] + boxsize[0] + steplength, steplength)
    y = np.arange(startpoint[1], startpoint[1] + boxsize[1] + steplength, steplength)
    z = np.arange(startpoint[2], startpoint[2] + boxsize[2] + steplength, steplength)
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


def plot_xPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,x=0,plot=[True,True,True],levelNum = 50):

    
    Y = np.arange(0,BOX_SIZE[1]+1,VOLUME_RESOLUTION)
    Z = np.arange(0,BOX_SIZE[2]+1,VOLUME_RESOLUTION)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharex=True, sharey=True)

    #Plot B_x
    if plot[0] == True:
        B_x = targetVolume[x,:,:,0].T
        
        
        title = "$\mathcal{B}_x$ at $X = $" + str(x)
        xlabel = "Y (cm)"
        ylabel = "Z (cm)"
        

        maxi = np.max(B_x+0.00001)
        mini = np.min(B_x-0.00001)
        CS = ax1.contourf(Y,Z,B_x,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        ax1.set_title(title)


    #Plot B_y
    if plot[1] == True:
        B_y = targetVolume[x,:,:,1].T
        
        
        title = "$\mathcal{B}_y$ at $X = $" + str(x)
        xlabel = "Y (cm)"
        ylabel = "Z (cm)"
        
        maxi = np.max(B_y+0.00001)
        mini = np.min(B_y-0.00001)
        CS = ax2.contourf(Y,Z,B_y,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        
        ax2.set_title(title)


    #plot B_z
    if plot[2] == True:
        B_z = targetVolume[x,:,:,2].T 
        
        title = "$\mathcal{B}_z$ at $X = $" + str(x)
        xlabel = "Y (cm)"
        ylabel = "Z (cm)"

        maxi = np.max(B_z+0.00001)
        mini = np.min(B_z-0.00001)
        CS = ax3.contourf(Y,Z,B_x,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        
        ax3.set_title(title)


    axlist = [ax1,ax2,ax3]    
    fig.colorbar(CS, ax=axlist)
    

    plt.show()


def plot_yPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,y=0,plot=[True,True,True],levelNum = 50):

    
    X = np.arange(0,BOX_SIZE[0]+1,VOLUME_RESOLUTION)
    Z = np.arange(0,BOX_SIZE[2]+1,VOLUME_RESOLUTION)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharex=True, sharey=True)

    #Plot B_x
    if plot[0] == True:
        B_x = targetVolume[:,y,:,0].T
        
        
        title = "$\mathcal{B}_x$ at $Y = $" + str(y)
        xlabel = "X (cm)"
        ylabel = "Z (cm)"

        maxi = np.max(B_x+0.00001)
        mini = np.min(B_x-0.00001)
        CS = ax1.contourf(X,Z,B_x,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        ax1.set_title(title)


    #Plot B_y
    if plot[1] == True:
        B_y = targetVolume[:,y,:,1].T
        
        
        title = "$\mathcal{B}_y$ at $Y = $" + str(y)
        xlabel = "X (cm)"
        ylabel = "Z (cm)"

        maxi = np.max(B_y+0.00001)
        mini = np.min(B_y-0.00001)
        CS = ax2.contourf(X,Z,B_y,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        
        ax2.set_title(title)


    #plot B_z
    if plot[2] == True:
        B_z = targetVolume[:,y,:,2].T 
        
        title = "$\mathcal{B}_z$ at $Y = $" + str(y)
        xlabel = "X (cm)"
        ylabel = "Z (cm)"
        
        maxi = np.max(B_z+0.00001)
        mini = np.min(B_z-0.00001)
        CS = ax3.contourf(X,Z,B_z,cmap=cm.magma, levels = np.linspace(mini,maxi,levelNum))

        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        
        
        ax3.set_title(title)


    axlist = [ax1,ax2,ax3]    
    fig.colorbar(CS, ax=axlist)
    

    plt.show()


def plot_zPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,z=0,plot=[True,True,True],levelNum = 50):

    X = np.arange(0,BOX_SIZE[0]+1,VOLUME_RESOLUTION)
    Y = np.arange(0,BOX_SIZE[1]+1,VOLUME_RESOLUTION)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharex=True, sharey=True)

    #Plot B_x
    if plot[0] == True:
        B_x = targetVolume[:,:,z,0].T
        
        
        title = "$\mathcal{B}_x$ at $Z = $" + str(z)
        xlabel = "X (cm)"
        ylabel = "Y (cm)"

        maxi = np.max(B_x+0.00001)
        mini = np.min(B_x-0.00001)

        CS = ax1.contourf(X,Y,B_x,cmap=cm.magma,levels= np.linspace(mini,maxi,levelNum))

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        ax1.set_title(title)


    #Plot B_y
    if plot[1] == True:
        B_y = targetVolume[:,:,z,1].T
        
        
        title = "$\mathcal{B}_y$ at $Z = $" + str(z)
        xlabel = "X (cm)"
        ylabel = "Y (cm)"

        maxi = np.max(B_y+0.00001)
        mini = np.min(B_y-0.00001)

        CS = ax2.contourf(X,Y,B_y,cmap=cm.magma,levels= np.linspace(mini,maxi,levelNum))

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        
        ax2.set_title(title)

    #plot B_z
    if plot[2] == True:
        B_z = targetVolume[:,:,z,2].T 
        
        title = "$\mathcal{B}_z$ at $Z = $" + str(z)
        xlabel = "X (cm)"
        ylabel = "Y (cm)"
        

        maxi = np.max(B_z+0.00001)
        mini = np.min(B_z-0.00001)

        CS = ax3.contourf(X,Y,B_z,cmap=cm.magma,levels= np.linspace(mini,maxi,levelNum))

        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        
        ax3.set_title(title)


    axlist = [ax1,ax2,ax3]    
    fig.colorbar(CS, ax=axlist)
    

    plt.show()


## 
def test():
    import tkinter as tk
    from tkinter import filedialog
    
    try:
        window = tk.Tk()
        input_filename = filedialog.askopenfilename(initialdir = "~/Desktop",
            title = "Select file containing coil geometry",
            filetypes = (("text file", "*.txt"),("all files","*.*")))
        
        output_filename = filedialog.asksaveasfilename(initialdir = "~/Desktop",
            title = "Select file to save to (*.npy binary)")
        window.destroy()
    except FileNotFoundError: pass
    
    BOX_SIZE = (30, 15, 15) # dimensions of box in cm (x, y, z)
    START_POINT = (0, -5, -5) # where the bottom left corner of the box is w/r to the coil coordinate system.
    
    COIL_RESOLUTION = 1 # cm; affects runtime of calculation process linearly, and increases precision up to a point
    VOLUME_RESOLUTION = 1 # cm; affects runtime of calculation process in n^3, and size of resulting Target Volume
    
    writeTargetVolume(input_filename,output_filename, BOX_SIZE,START_POINT, COIL_RESOLUTION, VOLUME_RESOLUTION)
    # writes result of calculation to file
    print("B-field output written to {}".format(output_filename))
    
    targetVolume = readTargetVolume(output_filename)
    
    print("Target volume loaded with shape:",targetVolume.shape)
    
    plot_xPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,x=0,plot=[True,True,True],levelNum = 50)
    plot_yPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,y=0,plot=[True,True,True],levelNum = 50)
    plot_zPlane(targetVolume,BOX_SIZE,VOLUME_RESOLUTION,z=0,plot=[True,True,True],levelNum = 50)
    
    
    # plot the coil
    coil_points = parseCoil(input_filename)
    fig = plt.figure()
    tick_spacing = 2
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("coil geometry")
    ax.set_xlabel("$X$ (cm)")
    ax.set_ylabel("$Y$ (cm)")
    ax.set_zlabel("$Z$ (cm)")
    ax.plot3D(*coil_points[:3],lw=2)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.show()

test()