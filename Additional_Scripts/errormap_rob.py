"""
Authors: I. Beslic & J. den Brok
date: 10.10.2019

Original IDL code from E. Rosolowski

NAME:
  	errmap_rob
PURPOSE:
   	To generate a map of the errors pixel-wise in a data cube by
   	taking the RMS along a pixel and then rejecting anything over
   	a threshold determined by the number of channels in a
   	spectrum, so that there is 25% chance of noise producing a spike
  	above the threshold.
"""



import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from scipy import stats
from astropy.io import fits
import matplotlib.pyplot as plt

def erf0(x, erftarg):
    """
    :param x: x-sigma interval of which erf gives us the probability that noise is in range [-x,x]
    :param erftarg: targeted error function value
    """
    return abs(erf(x)-erftarg)

def errmap_rob(file_to_cube):
    """
    :param file_to_cube: string path to the cube file
    """
    data, header = fits.getdata(file_to_cube, header=True)
    cube_dim = np.shape(data)
    
    # If cube does not have 3 dimensions, cannot do analysis, so return with error message
    if len(cube_dim)<3:
        raise IndexError("Holly Moly, Cube not 3-D! Try again.")

    if len(cube_dim)==4:
        if cube_dim[0]==1:
            data = data[0]
            cube_dim = np.shape(data)
        else:
            raise IndexError("Do not understand cube dimesnions, please provide cube in shape (z,x,y)")
    channels = cube_dim[0]
    x, y = cube_dim[1],cube_dim[2] 
    # Initial guess (taken from errmap_rob.pro)
    x0 = 3
    # The targeted errf is defined like that by E. Rosollowski, we don't knwo why
    #print(channels)
    args = 1-(5-1)/channels
    
    sig_false = minimize(erf0, x0, args=(args)).x
    
    
    cube_map = np.zeros((x, y)) + stats.median_absolute_deviation(data,axis=0)
    
    for i in range(x):
        
        for j in range(y):
            spec = data[:,i,j]
            
            # spec_neg: part of spectrum, where values are negative => Sure to only include noise
            # The following line is just a fancy way of ignoring the nans. It does the same as
            # spec_neg = spec[spec<0], but doesn't produce the runtime error warning
            
            spec_neg = spec[np.less(spec, 0., where=~np.isnan(spec))&  ~np.isnan(spec)] 
            
            if len(spec_neg) < 10:
                continue
            
            sigma = np.nanstd(np.concatenate((spec_neg, -spec_neg)))
            # simple code: cube_map[i,j] = np.nanstd(spec[spec<sig_false*sigma])
            cube_map[i,j] = np.nanstd(spec[np.less(spec, sig_false*sigma, where=~np.isnan(spec))])
            
    return cube_map

"""
#-------------------------------------------------------
#Example on how to save the errormap as an txt file:
#-------------------------------------------------------

filename="/vol/alcina/data1/jdenbrok/Proj_I_2019/data/ngc5194_co10.fits"

err_map = errmap_rob(filename)
np.savetxt("/users/jdenbrok/Desktop/test_python.txt", err_map)

#--------------------------------------
# Example on how to show the errormap
#--------------------------------------

filename="/vol/alcina/data1/jdenbrok/Proj_I_2019/data/ngc5194_co10.fits"
plt.imshow(errmap_rob(filename))
plt.colorbar()
plt.show()
"""
