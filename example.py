import biot_savart_v4_3 as bs

'''
Basic example using coil.txt
'''

bs.write_target_volume("coil.txt", "targetvol", (30, 15, 15), (-5, -5, -2.5), 1, 1)
# generates a target volume from the coil stored at coil.txt
# uses a 30 x 15 x 15 bounding box, starting at (-5, -0.5, -2.5)
# uses 1 cm resolution

bs.plot_coil("coil.txt")
# plots the coil stored at coil.txt

volume = bs.read_target_volume("targetvol")
# reads the volume we created

bs.plot_fields(volume, (30, 15, 15), (-5, -5, -2.5), 1, which_plane='z', level=1.5, num_contours=50)
# plots the fields we just produced, feeding in the same box size and start points.
# plotting along the plane x = 5, with 50 contours


'''
Making a pair of Helmholtz Coils
'''

bs.helmholtz_coils("helm1.txt", "helm2.txt", 50, 5, 2, 1)
# makes a pair of helmholtz coils
# 50 segments each, with radius of 5 cm
# spaced out by 2 cm, located at z = +/- 1 respectively
# 1 amp of current

bs.plot_coil("helm1.txt", "helm2.txt")

bs.write_target_volume("helm1.txt", "targetvol1", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)
bs.write_target_volume("helm2.txt", "targetvol2", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)
# use a target volume of size 10, centred about origin

h1 = bs.read_target_volume("targetvol1")
h2 = bs.read_target_volume("targetvol2")
# produce the target volumes we want

# use linear superposition of magnetic fields, to get the combined effects of multiple coils
h_total = h1 + h2

bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='z', level=0, num_contours=50)