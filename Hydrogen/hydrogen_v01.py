# Hydrogen v0.1
# Mingde Yin July 15, 2020

import numpy as np

'''
Trajectory
'''
targetvolume = np.ones((30, 15, 15)) # x is beam direction, z is B field, y is horizonal transverse

START_POINT = (0, 7, 7) # start at centre of one end

VEL_VECTOR = (1, 0 ,0)

VEL_VECTOR = VEL_VECTOR / np.linalg.norm(VEL_VECTOR)

SPEED = 10

STEP_NUMBER = 100

PASS_POINTS = np.linspace(START_POINT, START_POINT+VEL_VECTOR*30, STEP_NUMBER)

B = 0

for coord in PASS_POINTS:
    B.append(targetvolume[coord]) # this is basically the idea

'''
Calculation
'''

def H():
    # -mu dot B
    # using the B's from above
    times = 0 # somehow relate the times with the positions