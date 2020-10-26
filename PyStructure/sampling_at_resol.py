import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os.path
from os import path
import copy
from spectral_cube import SpectralCube
import radio_beam
from scipy import stats
from astropy import units as u
from reproject import reproject_interp
from astropy.wcs import WCS
from astropy.stats import median_absolute_deviation
from spectral_cube import SpectralCube
from twod_header import *
import radio_beam
import warnings
warnings.filterwarnings("ignore")

from gauss_conv import *

def sample_at_res(in_data,
                  ra_samp,
                  dec_samp,
                  in_hdr = None,
                  target_res_as = None,
                  target_hdr = None,
                  coverage = None,
                  rms = None,
                  show = False,
                  #advanced parameters
                  line_name = "",
                  galaxy = "",
                  save_fits = False,
                  path_save_fits = "",
                  perbeam = False):

    """
    Function to sample the data and convolve
    """

    #--------------------------------------------------------------
    #   Defaults and Definitions
    #-------------------------------------------------------------

    # Check if ra and dec arrays have same length
    if len(ra_samp) != len(dec_samp):
        print("[ERROR]\t Need matching RA and Dec vector. Returning")
        return np.nan

    # Check if data given as string (need to load data first) or already given
    # as an array.
    if isinstance(in_data, str):
        # Data given in form of a string, need to load data
        if not path.exists(in_data):
            print("[ERROR]\t File "+in_data+" not found. Returning. ")
            return ra_samp * np.nan
        data, hdr = fits.getdata(in_data, header = True)

    else:
        data = copy.deepcopy(in_data)
        if not in_hdr is None:
            hdr = in_hdr
        else:
            print("[ERROR]\t Provide a valid header with the data")
            return ra_samp*np.nan

    # Check if target resolution was provided
    if target_res_as is None:
        print("[WARNING]\t No target resolution specified.\n\t \
              Taking this as zero and sampling at the native resolution.")
        target_res_as = 0

    if target_hdr is None:
        print("[WARNING]\t No target header. Will use native astrometry")


    dim_data = np.shape(data)
    if len(dim_data) == 3:
        is_cube = True
    else:
        is_cube = False
    
    if not is_cube:
        target_hdr = twod_head(target_hdr)
    #--------------------------------------------------------------
    #   Convolve and Align
    #------------------------------------------------------------
   
    current_bmaj = hdr["BMAJ"]
    if current_bmaj < target_res_as/3600:
        data, hdr_out = conv_with_gauss(in_data= data, in_hdr = hdr,
                                              target_beam = target_res_as*np.array([1,1,0]),
                                              quiet = True,
                                              perbeam = perbeam)
        
        #data_speccube = SpectralCube(data=data, wcs = WCS(hdr))
        #data_speccube.beam = radio_beam.Beam(major=current_bmaj*u.arcsec, minor=current_bmaj*u.arcsec, pa=0*u.deg)
        #beam = radio_beam.Beam(major=target_res_as*u.arcsec, minor=target_res_as*u.arcsec, pa=0*u.deg)
        #new_datacube = data_speccube.convolve_to(beam)
        #data = np.array(new_datacube.unmasked_data[:,:,:])
        #hdr_out = hdr
    else:
        print("[INFO]\t Already at target resolution.")
        hdr_out = copy.copy(hdr)


    #; Measure the rms in the map after convolution but before
    #; alignment. This isn't really ideal but can be a useful
    #; shorthand for the noise.

    rms = median_absolute_deviation(data, axis = None,ignore_nan=True)


    # Align, if needed

    
    if not target_hdr is None:
    
        wcs_target = WCS(target_hdr)
        data_out, footprint = reproject_interp((data,hdr_out), target_hdr)
        data = data_out
        
        
        print("")
        
        # Save the convloved file as a fits file
        if save_fits:
            out_header = copy.copy(target_hdr)
            out_header["BMAJ"]=target_res_as/3600
            out_header["BMIN"]=target_res_as/3600
            out_header["LINE"]=line_name
            fits.writeto(path_save_fits+galaxy+'_'+str(line_name)+'_{}as.fits'.format(target_res_as), data=data_out, header=out_header, overwrite=True)
            print("[INFO]\t Convolved Fits file has been saved.")
        
    else:
        print("[INFO]\t No alignment because no target header supplied.")

    #--------------------------------------------------------------
    #   Sample
    #------------------------------------------------------------
    wcs_target = WCS(target_hdr)
    if is_cube:
        pixel_coords = wcs_target.all_world2pix(np.column_stack((ra_samp, dec_samp, np.zeros(len(dec_samp)))),0)
    else:
        pixel_coords = wcs_target.all_world2pix(np.column_stack((ra_samp, dec_samp)),0)
    samp_x = np.array(np.rint(pixel_coords[:,0]),dtype=int)
    samp_y = np.array(np.rint(pixel_coords[:,1]),dtype= int)
    
    
    
    n_pts = len(samp_x)
    dim_data = np.shape(data)


    if is_cube:
        result = np.zeros((n_pts, dim_data[0]))*np.nan
    else:
        result = np.zeros(n_pts)*np.nan

    coverage = np.zeros(n_pts)
    if is_cube:
        
        in_map = np.where((samp_x>0)& (samp_x <dim_data[2]) &
                          (samp_y > 0)& (samp_y < dim_data[1]))[0]
        in_map_ct = len(in_map)
    else:
        in_map = np.where((samp_x>0)& (samp_x <dim_data[1]) &
                          (samp_y > 0)& (samp_y < dim_data[0]))[0]
        in_map_ct = len(in_map)
        
    
    if in_map_ct>0:
        if is_cube:
            for kk in range(in_map_ct):
                
                result[in_map[kk],:] = data[:, samp_y[in_map[kk]],samp_x[in_map[kk]]]
                coverage[in_map[kk]] =  np.nansum(np.isfinite(data[:, samp_y[in_map[kk]],samp_x[in_map[kk]]]))>=1
            coverage = np.array(coverage, dtype = int)
        else:
            result[in_map] = data[samp_y[in_map],samp_x[in_map]]
            coverage[in_map] = np.isfinite(data[samp_y[in_map],samp_x[in_map]])

    #--------------------------------------------------------------
    #   Return
    #------------------------------------------------------------

    return result
