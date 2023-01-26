import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import copy

from twod_header import *
from create_hex_grid import *
def make_sampling_points(ra_ctr, dec_ctr,max_rad, spacing,mask, hdr_mask,
                     overlay_in = None, overlay_hdr_in = None, show = False):


    """

    ;science tool
    ;
    ;CALLING SEQUENCE:
    ;
    ; samp_ra, samp_dec = make_sampling_points $
    ;   , ra_ctr = ra_ctr
    ;   , dec_ctr = dec_ctr
    ;   , max_rad = max_rad
    ;   , spacing = spacing
    ;   , mask = in_mask
    ;   , hdr_mask = in_mask_hdr
    ;   , overlay = in_overlay
    ;   , hdr_overlay = in_overlay_hdr
    ;   , show = True
    ;
    ;
    ; INPUTS:
    ;
    ; ra_ctr, dec_ctr - center of the grid
    ;
    ; max_rad - maximum extent of grid in degrees. Default 1 degree.
    ;
    ; spacing - hex grid spacing.
    ;
    ; mask - string or image holding a 1s and 0s mask. Only points that
    ;        sample a pixel with a 1 are valid and kept. Points outside
    ;        the mask are also dropped. Three dimensional data are
    ;        collapsed with "or" (i.e., any 1 along the third axis is a 1
    ;        in the image).
    ;
    ; mask_hdr - if a file is not supplied, this holds a header that
    ;            supplies the mask astrometry.
    ;
    ; overlay, overlay_hdr - similar to mask but an image used only to
    ;                        display the data. Peak intensity is used for
    ;                        three dimensional data.
    ;
    ; show - flag that sets whether an image with the points is displayed.
    ;
    ; OUTPUTS:
    ;
    ; samp_ra, samp_dec - return arrays of RA, Dec sampling
    ;                     points. Returns not-a-number if it fails.
    ;
    ; MODIFICATION HISTORY:
    ; Converted to Python by J. den Brok, 16 Oct 2019
    ;-


    """

    #--------------------------------------------------------------
    #  DEFAULTS AND DEFINITIONS
    #--------------------------------------------------------------

    n_dim_mask = len(np.shape(mask))

    #Collaps the mask to two dimensions,, if needed
    if n_dim_mask == 3:
        print("[INFO]\t Collapsing mask tp two dimensions.")
        mask = np.sum(np.isfinite(mask), axis = 0)>=1
        mask_hdr = twod_head(hdr_mask)
    
    mask_dim = np.shape(mask)

    wcs = WCS(hdr_mask)
    
    #--------------------------------------------------------------
    #  Check if r_max needs to be computed
    #--------------------------------------------------------------
    if max_rad in ["auto"]:
        from astropy.coordinates import SkyCoord
        #find coordinates of bottom left and top right corner of image
        dx = mask_dim[1]
        dy = mask_dim[0]
        
        c_1 = SkyCoord.from_pixel(0,0, wcs)
        c_2 = SkyCoord.from_pixel(dx,dy, wcs)
        
        #determine length of the diagonal
        max_rad = c_1.separation(c_2).value/2
        
        print("[INFO]\t Overlay Size set to {}deg".format(np.round(max_rad, 3)))
    #--------------------------------------------------------------
    #  Generate a hexagrid
    #--------------------------------------------------------------

    samp_ra, samp_dec = hex_grid(ra_ctr, dec_ctr, spacing, radec = True, r_limit = max_rad)

    #--------------------------------------------------------------
    #  Pare to desired scope based on input map
    #--------------------------------------------------------------

    
    try:
        pixel_coords = wcs.all_world2pix(np.column_stack((samp_ra, samp_dec)),0)
    except:
        pixel_coords = wcs.all_world2pix(np.column_stack((samp_ra, samp_dec, np.zeros(len(samp_ra)))),0)
    samp_x = np.array(np.rint(pixel_coords[:,0]), dtype=int)
    samp_y = np.array(np.rint(pixel_coords[:,1]), dtype=int)


    keep = np.where(np.dot(samp_x>=0,1) & np.dot(samp_y>=0,1) &
                    np.dot(samp_x<mask_dim[1],1) & np.dot(samp_y<mask_dim[0],1))[0]
    keep_ct = len(keep)

    if keep_ct == 0:
        print("[ERROR]\t No sampling points survive inside mask, returning NaNs.")
        return np.nan, np.nan

    samp_ra = samp_ra[keep]
    samp_dec = samp_dec[keep]
    samp_x = samp_x[keep]
    samp_y = samp_y[keep]



    keep = np.where(mask[samp_y, samp_x])
    keep_ct = len(keep[0])

    if keep_ct == 0:
        print("[ERROR]\t No sampling points survive inside mask, returning NaNs.")
        return np.nan, np.nan

    samp_ra = samp_ra[keep]
    samp_dec = samp_dec[keep]
    samp_x = samp_x[keep]
    samp_y = samp_y[keep]


    #--------------------------------------------------------------
    #  Visualize, if required
    #--------------------------------------------------------------

    if show:
        have_overlay = False

        #check, if overlay given as path
        if not overlay_in is None:
            if isinstance(overlay_in, str):
                overlay, overlay_hdr = fits.getdata(overlay_in, header = True)

            else:
                overlay = copy.deepcopy(overlay_in)
                if not overlay_hdr_in is None:
                    overlay_hdr = overlay_hdr_in
            have_overlay = True

            dim_overlay =np.shape(overlay)
            if len(dim_overlay)==3:
                #overlay = np.nanmax(overlay, 0)
                overlay = np.nansum(overlay, 0)
                overlay_hdr = twod_head(overlay_hdr)

        if not have_overlay:
            overlay = copy.deepcopy(mask)
            overlay_hdr = hdr_mask
            have_overlay=True

        if have_overlay:
            wcs_overlay = WCS(overlay_hdr)
            pixel_coords_ov = wcs_overlay.all_world2pix(np.column_stack((samp_ra, samp_dec)),0)
            samp_x_ov = np.array(pixel_coords_ov[:,0])
            samp_y_ov = np.array(pixel_coords_ov[:,1])

            plt.figure()
            plt.plot(samp_x_ov,samp_y_ov, "h", markersize = 16)
            #plt.contour(samp_x_ov,samp_y_ov,mask, levels=[1], colors="k")
            plt.show()


    return samp_ra, samp_dec

"""
filename = '/vol/alcina/data1/jdenbrok/Proj_I_2019/ngc_5194_database/working_data/iram/ngc5194_co21.fits'
hcn_cube, header =  fits.getdata(filename, header = True)

mask = np.sum(np.isfinite(hcn_cube), axis = 0)>=1
mask_hdr = twod_head(header)

spacing = 40/3600./2.
test_ra, test_dec = make_sampling_points(202.4696292, 47.1951722, 1., spacing, mask, mask_hdr, show = True)
"""
