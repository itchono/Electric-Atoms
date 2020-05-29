import numpy as np
import numpy.linalg as lg
import cProfile
x = np.arange(-15,15,1)
y = np.arange(-15,15,1)
z = np.arange(-15,15,1)
X,Y,Z = np.meshgrid(x, y, z)
N = 1000
target_points = np.random.normal(0,3,3*N).reshape(N,3) 
def speed_test():
    for target_point in target_points:
        tx,ty,tz = target_point
        Rvec = np.array([(X-tx),(Y-ty),(Z-tz)])
        R = lg.norm(Rvec,axis=0) 
        # alternatively:
        #  R = np.sqrt( (X-tx)**2 + (Y-ty)**2 + (Z-tz)**2 )
        F = 1/R**3
    print(R.shape)
    print(F.shape)
cProfile.run("speed_test()")