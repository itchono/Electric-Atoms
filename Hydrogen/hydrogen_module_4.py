# Hydrogen
# Oracle Field Producer (4)

# Other options: Analytical Fit

import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

x = np.linspace(0, 4, 5)
y = np.linspace(0, 3, 4)
z = np.linspace(0, 4, 5)

def test_field(x, y, z):
    return x+y+z

data = test_field(*np.meshgrid(x, y, z, indexing='ij'))


def oracle(atoms, Bfields, offset, spacing: float):
    '''
    Returns the (Bx, By, Bz) fields over time using a nearest neighbour approximation
    '''
    result_fields = []

    print(Bfields)

    X,Y,Z = Bfields.shape

    x = np.linspace(offset[0], offset[0] + X - spacing, X)
    y = np.linspace(offset[1], offset[1] + Y - spacing, Y)
    z = np.linspace(offset[2], offset[2] + Z - spacing, Z)

    magnetic_field = rgi((x, y, z), Bfields)

    for atom in atoms:
        positions = atom.T / spacing + offset # apply snapping to grid

        
        result_fields.append(magnetic_field(positions))

    return result_fields

bee = oracle(np.array([[[0, 0.5, 0.7], [0, 1.5, 1.8], [0, 1.7, 1.8]]]), data, np.array([0, 0, 0]), 1)
print(bee)