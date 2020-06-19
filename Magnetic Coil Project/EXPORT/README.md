# Biot-Savart Magnetic Field Calculator

This tool numerically solves for the magnetic field around an arbitrarily shaped coil specified by the user, in a discrete and finite volume surrounding the coil.

Latest Version: V4.2 (June 18, 2020)

# Basic Overview of Functionality
Given an input coil, the tool can both calculate, and plot the magnetic field vector over a finite volume of space around the coil.

The coil is passed in as a set of vertices describing the geometry of the coil, as well as the amount of current flowing through a given segment of the coil. (See Formatting section)

The c


# Installation and Usage
### STAGE 1: Code Installation and Prerequisites
* Have Python 3.x installed
* Install `numpy` and `matplotlib`
* Clone or download the repository to your c








# Changelog
* v1: Initial Release

* v2: Code accelerated using numpy meshgrids
* v2.1: Tkinter dialogs for opening & saving files. Defaults of 1 cm resolution in calculation.
* v3: Plotting code integrated.
* v3.1: Minor cosmetic improvements to plot.
* v3.2: 3D plot of coil geometry.
* v3.3/3.4: Plotted B-fields together but code is long.
* v3.5/3.6: all B-field plots together
* v3.7: B-fields plotted together with 50 levels (now works on windows) and combined v3.3 and v3.5
* v3.8: Changed up all np.aranges to np.linspaces and changed up the plotting 
code to work with non-integer step sizes and non-integer levels
* v4: Using Richardson Extrapolation for midpoint rule to improve accuracy (5 to 30x better at 1.4x speed penalty), tweaked linspaces to correctly do step size
* v4.1: Minor change in function indexing to use more numpy, cleaning up for export
* v4.2: Changed the linspaces a bit to make everything more symmetric