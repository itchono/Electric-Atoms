import biot_savartv4_2 as b4
import biot_savart_v2 as b2
import time
import numpy as np
import cProfile, pstats, io
from pstats import SortKey


if __name__ == "__main__":
    '''
    A little demo program which saves the coil's corresponding target volume to file, and lets you get the B vector at any point in the box.
    
    
    Results:
    Around 1.4x slower
    for around 5 to 30x precision boost
    
    '''

    # specify the volume over which the fields should be calculated
    BOX_SIZE = (30, 15, 15) # dimensions of box in cm (x, y, z)
    START_POINT = (-5, -2.5, -7.5) # bottom left corner of box w.r.t. coil coordinate system
    
    COIL_RESOLUTION = 1 # cm
    VOLUME_RESOLUTION = 0.5 # cm

    '''b2.writeTargetVolume("midpoint1", BOX_SIZE, START_POINT, 1, 0.5)
    print("Done")
    b2.writeTargetVolume("midpoint01", BOX_SIZE, START_POINT, 0.1, 0.5)
    print("Done")
    b2.writeTargetVolume("midpoint001", BOX_SIZE, START_POINT, 0.01, 0.5)
    print("Done")'''

    b2.writeTargetVolume("midpoint05", BOX_SIZE, START_POINT, 0.67, 0.5)
    reference001 = b4.readTargetVolume("midpoint001")
    reference01 = b4.readTargetVolume("midpoint01")
    reference1 = b4.readTargetVolume("midpoint1")
    reference05 = b4.readTargetVolume("midpoint05")

    '''b4.writeTargetVolume("coil.txt", "richardson1", BOX_SIZE, START_POINT, volumeresolution=0.5)
    print("Done")
    b4.writeTargetVolume("coil.txt", "richardson01", BOX_SIZE, START_POINT, coilresolution=0.1, volumeresolution=0.5)
    print("Done")'''

    richardson1 = b4.readTargetVolume("richardson1")
    richardson01 = b4.readTargetVolume("richardson01")

    deviationr1 = reference1 - reference001
    deviationr1r1 = richardson1 - reference001

    b4.plot_fields(deviationr1, START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=1.25)

    b4.plot_fields(reference05 - reference001, START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=1.25)

    b4.plot_fields(deviationr1r1, START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=1.25)

    b4.plot_fields(reference01 - reference001, START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=1.25)

    b4.plot_fields(richardson01 - reference001, START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=1.25)
    
