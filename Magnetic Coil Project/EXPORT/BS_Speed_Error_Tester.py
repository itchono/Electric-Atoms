import biot_savart_v4_dev as b4d
import biot_savart_v4 as b4
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
    reference001 = b4.readTargetVolume("midpoint001")
    reference01 = b4.readTargetVolume("midpoint01")
    reference1 = b4.readTargetVolume("midpoint1")

    richardson1 = b4.readTargetVolume("richardson1")
    richardson01 = b4.readTargetVolume("richardson01")

    POINT = (-5, -2.5, -7.5)

    boi001 = b4.getFieldVector(reference001, POINT, (-5, -2.5, -7.5), 1)
    boi01 = b4.getFieldVector(reference01, POINT, (-5, -2.5, -7.5), 1)
    boi1 = b4.getFieldVector(reference1, POINT, (-5, -2.5, -7.5), 1)
    boir1 = b4.getFieldVector(richardson1, POINT, (-5, -2.5, -7.5), 1)
    boir01 = b4.getFieldVector(richardson01, POINT, (-5, -2.5, -7.5), 1)

    print(boi001, boi01, boi1, boir1, boir01)

    print("Error with each function")
    print("Midpoint, coil res = 1 cm", boi001 - boi1)
    print("2 Stage Richardson, coil res = 1 cm", boi001 - boir1)
    print("Midpoint, coil res = 0.1 cm", boi001 - boi01)
    print("2 Stage Richardon, coil res = 0.1 cm", boi001 - boir01)

    print("Midpoint, coil res = 1 cm", np.linalg.norm(reference001 - reference1))
    print("2 Stage Richardson, coil res = 1 cm", np.linalg.norm(reference001 - richardson1))
    print("Midpoint, coil res = 0.1 cm", np.linalg.norm(reference001 - reference01))
    print("2 Stage Richardon, coil res = 0.1 cm", np.linalg.norm(reference001 - richardson01))

    print("Speed test")

    pr = cProfile.Profile()
    pr.enable()

    b4.writeTargetVolume("coil.txt","yes", 
                    (30, 15, 15),(-5, -2.5, -7.5),1,1)


    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


    pr = cProfile.Profile()
    pr.enable()

    b4d.writeTargetVolume("coil.txt","yes", 
                    (30, 15, 15),(-5, -2.5, -7.5),1,1)


    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    pr = cProfile.Profile()
    pr.enable()

    b2.writeTargetVolume("yes", 
                    (30, 15, 15),(-5, -2.5, -7.5),1,1)


    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    
