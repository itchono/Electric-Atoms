from biot_savartv4_2 import *




def segmented_circle_test(numSegments=36):

    f = open("coil.txt","w")
    line = ""
    for i in range(0,numSegments,1):
        line = str(np.cos(2*np.pi*(i)/(numSegments-1))*5) + "," + str(np.sin(2*np.pi*(i)/(numSegments-1))*5) + ",0,5\n"
        f.write(line)
    f.close()



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
    
    # specify the volume over which the fields should be calculated
    BOX_SIZE = (10, 10, 10) # dimensions of box in cm (x, y, z)
    START_POINT = (-5, -5, -5) # bottom left corner of box w.r.t. coil coordinate system
    
    COIL_RESOLUTION = 0.5 # cm
    VOLUME_RESOLUTION = 0.5 # cm

    # save result of calculation to file
    writeTargetVolume(input_filename,output_filename, 
                    BOX_SIZE,START_POINT,COIL_RESOLUTION,VOLUME_RESOLUTION)
    print("B-field output written to {}".format(output_filename))
    
    # read in computed data 
    Bfields = readTargetVolume(output_filename)
    print("Calculated B-fields loaded. Array shape:",Bfields.shape)
    #print(Bfields)

    # plot B-fields
    plot_fields(Bfields,START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=0,num_contours=50)
        
    # plot the coil geometry
    plot_coil(input_filename)

segmented_circle_test()
'''
It Seems that the segmented circle does in fact work as intended, a decent amount of modifications where made to the original code as I found a few different 
"glitches" when it comes to assigning the magnetic field to the actual points on the plots (some weird scaling issue) and another problem when using non-integer step
sizes for the volume (use np.linspace instead of np.arange).

The B_z field is indeed circular and varies with the radius, the center having the strongest field, not exactly uniform. One weird thing is the B_x and B_y plots,
theoretically they should both be 0, maybe the result of some weird fringing? Not too sure. B_x seems to be a function of the Y coordinate and B_y seems to be a function
of the x coordinate, they both seem like the fields produced by some weird point charge. Must be investigated further.
'''

def helmholtz_test(numSegments=36):

    f = open("coil.txt","w")
    line = ""
    for i in range(0,numSegments,1):
        line = str(np.cos(2*np.pi*(i)/(numSegments-1))*5) + "," + str(np.sin(2*np.pi*(i)/(numSegments-1))*5) + ",0,5\n"
        f.write(line)
    f.close()



    f2 = open("coil2.txt","w")
    for i in range(0,numSegments,1):
        line = str(np.cos(2*np.pi*(i)/(numSegments-1))*5) + "," + str(np.sin(2*np.pi*(i)/(numSegments-1))*5) + ",2.5,5\n"
        f2.write(line)
    f2.close()




    import tkinter as tk
    from tkinter import filedialog
    
    try:
        window = tk.Tk()
        input_filename = filedialog.askopenfilename(initialdir = "~/Desktop",
            title = "Select file containing coil geometry",
            filetypes = (("text file", "*.txt"),("all files","*.*")))

        input_filename2 = filedialog.askopenfilename(initialdir = "~/Desktop",
            title = "Select file containing coil geometry",
            filetypes = (("text file", "*.txt"),("all files","*.*")))
        
        
        output_filename = filedialog.asksaveasfilename(initialdir = "~/Desktop",
            title = "Select file to save to (*.npy binary)")

        output_filename2 = filedialog.asksaveasfilename(initialdir = "~/Desktop",
            title = "Select file to save to (*.npy binary)")
        
        window.destroy()
    except FileNotFoundError: pass



    
    # specify the volume over which the fields should be calculated
    BOX_SIZE = (10, 10, 10) # dimensions of box in cm (x, y, z)
    START_POINT = (-5, -5, -5) # bottom left corner of box w.r.t. coil coordinate system
    
    COIL_RESOLUTION = 0.5 # cm
    VOLUME_RESOLUTION = 0.5 # cm

    # save result of calculation to file
    writeTargetVolume(input_filename,output_filename, 
                    BOX_SIZE,START_POINT,COIL_RESOLUTION,VOLUME_RESOLUTION)

    writeTargetVolume(input_filename2,output_filename2, 
                    BOX_SIZE,START_POINT,COIL_RESOLUTION,VOLUME_RESOLUTION)
    print("B-field output written to {} and {}".format(output_filename,output_filename2))
    
    # read in computed data 
    BfieldsA = readTargetVolume(output_filename)
    BfieldsB = readTargetVolume(output_filename2)

    #Using superposition principle
    Bfields = np.add(BfieldsA,BfieldsB)
    print("Calculated B-fields loaded. Array shape:",Bfields.shape)
    #print(Bfields)

    # plot B-fields
    plot_fields(Bfields,START_POINT,BOX_SIZE,VOLUME_RESOLUTION,which_plane='z',level=0,num_contours=50)
        
    # plot the coil geometry


    coil_points = parseCoil(input_filename)
    coil_points2 = parseCoil(input_filename2)
    fig = plt.figure()
    tick_spacing = 2
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_zlabel("$z$ (cm)")
    ax.plot3D(coil_points[0],coil_points[1],coil_points[2],lw=2)
    ax.plot3D(coil_points2[0],coil_points2[1],coil_points2[2],lw=2)


    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.show()



helmholtz_test()
'''
For this test case, I found the B fields on the Z plane right in between both Helmholtz coils. It worked as intended, but once again there was some weird B_x and B_y
magnetic fields, just like the above case with the single loop, but much less pronounced. Once again, might be a result of some weird fringing.

'''