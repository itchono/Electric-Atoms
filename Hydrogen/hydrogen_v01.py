# Hydrogen v0.1
# Mingde Yin July 15, 2020

import numpy as np


targetvolume = np.ones((30, 15, 15)) # x is beam direction, z is B field, y is horizonal transverse

START_POINT = (0, 7, 7) # start at centre of one end

VEL_VECTOR = (1, 0 ,0)

VEL_VECTOR = VEL_VECTOR / np.linalg.norm(VEL_VECTOR)

SPEED = 10



PASS_POINTS = 