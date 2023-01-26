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
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

from gauss_conv import *


def get_vaxis(hdr):
    v = np.arange(hdr["NAXIS3"])
    vdif = v-(hdr["CRPIX3"]-1)
    vaxis = vdif * hdr["CDELT3"] + hdr["CRVAL3"]
    return vaxis

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
                  perbeam = False,
                  spec_smooth = ["default","binned"]):

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
   
    

    rms = median_absolute_deviation(data, axis = None,ignore_nan=True)


    ######
    # Check that velocity axis is in m/s, not km/s
    ######
    if is_cube:
        if abs(target_hdr["CDELT3"])<200:
            print("[INFO]\t Overlay Cube in km/s, converting to m/s.")
            target_hdr["CDELT3"] = 1000 * target_hdr["CDELT3"]
            target_hdr["CRVAL3"] = 1000 * target_hdr["CRVAL3"]
            target_hdr["CUNIT3"] = "m/s"
            
        if abs(hdr["CDELT3"])<200:
            print("[INFO]\t Line Cube in km/s, converting to m/s.")
            hdr["CDELT3"] = 1000 * hdr["CDELT3"]
            hdr["CRVAL3"] = 1000 * hdr["CRVAL3"]
            hdr["CUNIT3"] = "m/s"
        
        #check if cube vaxis is inverted (want delta v>0):
        if target_hdr["CDELT3"]<0:
            print("[INFO]\t Target Cube has invertied vaxis. Re-inverting...")
            vaxis_inv = get_vaxis(target_hdr)
            target_hdr["CDELT3"] = -1* target_hdr["CDELT3"]
            target_hdr["CRPIX3"] = 1
            target_hdr["CRVAL3"] = vaxis_inv[-1]
            
        if hdr["CDELT3"]<0:
            print("[INFO]\t Line Cube has invertied vaxis. Re-inverting...")
            vaxis_inv = get_vaxis(hdr)
            hdr["CDELT3"] = -1* hdr["CDELT3"]
            hdr["CRPIX3"] = 1
            hdr["CRVAL3"] = vaxis_inv[-1]
            
            #flip the cube
            data = np.flip(data, axis=0)
            
    #perform spectral smooting, if needed
    if spec_smooth[0] in ["overlay"]:
        spec_smooth[0] = abs(target_hdr["CDELT3"])/1000
    if type(spec_smooth[0]) == int or type(spec_smooth[0]) == float:
        spec_res = abs(hdr["CDELT3"])/1000
        fwhm_factor = np.sqrt(8*np.log(2))
        if spec_res >=spec_smooth[0]:
            print("[INFO]\t No spectral smoothing; already at target resolution.")
        else:
            print("[INFO]\t Do spectral smoothing to {} km/s".format(spec_smooth[0]))
            
            
            #check the method:
            #
            # Gauss method:
            if spec_smooth[1] in ["gauss"]:
                pix = ((spec_smooth[0]**2 - spec_res**2)**0.5  / spec_res)/fwhm_factor
                kernel = Gaussian1DKernel(pix)
            
                
                for spec_n in ProgressBar(range(dim_data[1]*dim_data[2])):
                    y = spec_n%dim_data[1]
                    x = spec_n//dim_data[1]
                    data[:,y,x] = convolve(data[:, y,x],kernel)
          
                
            # Binning method
            elif spec_smooth[1] in ["binned", "combined"]:
                vaxis_native = get_vaxis(hdr_out)
                
                n_ratio = int(spec_smooth[0]/spec_res)
                
                #smooth, if remainder of target/res is almost 90%
                if (spec_smooth[0]/spec_res-n_ratio)>0.9:
                    n_ratio+=1
                new_len = len(vaxis_native)//n_ratio
                if n_ratio==1:
                    print("[INFO]\t No spectral smoothing; already at target resolution.")
                else:
                    new_vaxis = np.array([np.nanmean(vaxis_native[n_ratio*j:n_ratio*(j+1)]) for j in range(new_len)])
                    data = np.array([np.nanmean(data[n_ratio*j:n_ratio*(j+1),:,:], axis=0) for j in range(new_len)])
                 
                    hdr_out["NAXIS3"] = new_len
                    hdr_out["CDELT3"] = new_vaxis[1]-new_vaxis[0]
                    hdr_out["CRVAL3"] = new_vaxis[0] + (hdr_out["CRPIX3"]-1)*hdr_out["CDELT3"]
                
                if spec_smooth[1] in ["combined"]:
                    print(n_ratio*spec_res)
                    if n_ratio*spec_res<spec_smooth[0]:
                        pix = ((spec_smooth[0]**2 - (n_ratio*spec_res)**2)**0.5  / spec_res)/fwhm_factor
                        kernel = Gaussian1DKernel(pix)
            
                    
                        for spec_n in ProgressBar(range(dim_data[1]*dim_data[2])):
                            y = spec_n%dim_data[1]
                            x = spec_n//dim_data[1]
                            data[:,y,x] = convolve(data[:, y,x],kernel)
                   
            
            
    # Align, if needed

    
    trg_hdr = copy.deepcopy(target_hdr)
    #--------------------------------------------------------------
    #   Sample
    #------------------------------------------------------------
    wcs_target = WCS(trg_hdr)
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

    return result, trg_hdr
