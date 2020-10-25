import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def sphdist(lat1, long1, lat2, long2, degrees = False):
    """
    Copyright (C) 1991, Johns Hopkins University/Applied Physics Laboratory
    This software may be used, copied, or redistributed as long as it is not
    sold and this copyright notice is reproduced on each copy made.  This
    routine is provided as is without any express or implied warranties
    whatsoever.  Other limitations apply as described in the file disclaimer.txt.
    Converted to IDL V5.0   W. Landsman   September 1997

    Tranlsated to python, 18. Oct 2019, J. den Brok
    """
    cf = 1.0
    if degrees:
        cf = 180/np.pi

    rxy, z1 = pol2cart(1.0, lat1/cf)
    x1, y1 = pol2cart(rxy, long1/cf)
    rxy, z2 = pol2cart(1.0, lat2/cf)
    x2, y2 = pol2cart(rxy, long2/cf)

    #--- Compute vector dot product for both points. ---
    cs = x1*x2 + y1*y2 + z1*z2

    #--- Compute the vector cross product for both points. ---
    xc = y1*z2 - z1*y2
    yc = z1*x2 - x1*z2
    zc = x1*y2 - y1*x2
    sn = np.sqrt(xc*xc + yc*yc + zc*zc)

    r, a = cart2pol(cs, sn)
    return cf*a

def get_pixel_scale(hdr,tol = 1e-6):
    
    wcs = WCS(hdr)
    x = np.array([0,1,0])
    y = np.array([0,0,1])
    
    if hdr["NAXIS"]==3:
        phys_coords = wcs.all_pix2world(np.column_stack((x,y, np.zeros(3))),0)
    else:
        phys_coords = wcs.all_pix2world(np.column_stack((x,y)),0)
    ra = phys_coords[:,0]
    dec = phys_coords[:,1]

    step_x = abs(sphdist(ra[0], dec[0], ra[1], dec[1], True))
    step_y = abs(sphdist(ra[0], dec[0], ra[2], dec[2], True))
    
    if abs(step_x - step_y) >tol:
        print("[WARNING]\t Pixel scale looks different in X and Y.")
        return  np.sqrt(step_x*step_y)


    return step_x
