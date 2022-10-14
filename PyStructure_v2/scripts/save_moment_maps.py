"""
Author: L. Neumann + J. den Brok
"""
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import griddata
from reproject import reproject_interp
from astropy.io import fits
import copy

def sample_to_hdr(in_data,  # input data on hexagonal grid (ra_samp, dec_samp)
                  ra_samp,  # right ascension of hexagonal pixels
                  dec_samp, # declination of hexagonal pixels
                  in_hdr,  # header used to create the grid
                  ):
    """
    Function to regrid the data from hexagonal to rectangular grid
    specified by the header
    """
    
    
    #--------------------------------------------------------------
    #  Create Pixel Grid from Header
    #--------------------------------------------------------------
    
    # make x-axis array in pixel units
    x_axis = np.arange(in_hdr['NAXIS1'])

    # make y-axis array in pixel units
    y_axis = np.arange(in_hdr['NAXIS2'])

    # produce grid in pixel units
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    
    # load coordinate system
    wcs = WCS(in_hdr)

    # convert hexagonal coordiates to rectangular pixel units
    pixel_coords = wcs.all_world2pix(np.column_stack((ra_samp, dec_samp)),0)
    
    #--------------------------------------------------------------
    #  Sample Data onto Pixel Grid
    #--------------------------------------------------------------
    
    # sample data onto pixel grid
    data_grid = griddata(pixel_coords, in_data, (grid_x, grid_y), method='nearest')
    
    return data_grid
    
def resample_hdr(hdr_ov, target_res):
    """
    Function to resample the grid to account for lower target resolution
    :param target_res: The PyStructure resolution in arcsec
    """
    
    #create_wcs header file
    wcs_new = WCS(naxis=2)
    wcs_new.wcs.crpix = [1, 1]
    wcs_ov = WCS(hdr_ov)
    ra_ref, dec_ref = wcs_ov.all_pix2world(0,0 ,0)

    wcs_new.wcs.crval = [ra_ref, dec_ref]
    wcs_new.wcs.cunit = ["deg", "deg"]
    wcs_new.wcs.ctype = ["RA---TAN", "DEC--TAN"]


    delta_px = target_res/3600/3
    wcs_new.wcs.cdelt = [-delta_px, delta_px]
    
    #length of axes
    xaxis_n = int(np.round(hdr_ov['NAXIS1']*abs(hdr_ov['CDELT1'])/delta_px))
    yaxis_n = int(np.round(hdr_ov['NAXIS2']*abs(hdr_ov['CDELT2'])/delta_px))
    
    wcs_new.array_shape = [xaxis_n, yaxis_n]
    
    hdr_new = wcs_new.to_header()
    hdr_new["NAXIS"]=2
    hdr_new["NAXIS1"]=xaxis_n
    hdr_new["NAXIS2"]= yaxis_n
    
    hdr_new["BMAJ"] = target_res/3600
    hdr_new["BMIN"] = target_res/3600
    hdr_new["BPA"] = target_res/3600
    
    return hdr_new
    
    
def save_to_fits(ra,
                 dec,
                 hdr_in,
                 ov_slice,
                 key,
                 filename,
                 this_source,
                 this_data,
                 line,
                 folder,
                 target_res):

    data_in = copy.deepcopy(this_data["INT_"+key+"_"+line.upper()])
    
    map_cartesian = sample_to_hdr(data_in,
                                           ra,
                                           dec,
                                           hdr_in)
    #make edges to nan
    map_cartesian = ov_slice*map_cartesian
    #if resolution of overlay is below the target resolution, we need to resample
    if 3600*min([hdr_in["BMAJ"],hdr_in["BMIN"]])<0.99*target_res:
        hdr_in_repr = resample_hdr(hdr_in, target_res)
        #reproject the cartesian map using new
        map_cartesian, footprint = reproject_interp((map_cartesian,hdr_in), hdr_in_repr)
        hdr_in = hdr_in_repr
        
        
    fits.writeto(folder+this_source+"_"+line+"_"+filename+".fits", data =map_cartesian, header =  hdr_in, overwrite=True)
    
    
    

def save_mom_to_fits(fname,
                     lines_data,
                     source_list,
                     run_success,
                     target_hdr_list, target_slice_list, folder, target_res):
    """
    Function to prepare and convert the moment maps created on a hexagonal grid onto a cartesian one
    and save as FITS file
    :param fname: list of string with PyStructure file names
    :param lines_data: PD Database with information about individual spectral lines
    :param source_list: List of Strings; name of individual sources
    :param target_hdr_list: fits file 2D header of target (overlay image)
    """
    
    n_sources = len(source_list)
    n_lines = len(lines_data["line_name"])

    for ii in range(n_sources):

        #if the run was not succefull, don't do processing of the data
        if not run_success[ii]:
            continue
        
        this_source = source_list[ii]
            
        #load the PyStructure
        this_data = np.load(fname[ii],allow_pickle = True).item()
        
        #load the coordinates
        ra_deg = this_data["ra_deg"]
        dec_deg = this_data["dec_deg"]
        
        target_slice = target_slice_list[ii]
        target_slice[np.isfinite(target_slice)]=1
        
        for line in lines_data["line_name"]:
            #iterate over the moment maps
            #mom0:
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"VAL","mom0",this_source,this_data,line,folder,target_res)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"UC","emom0",this_source,this_data,line,folder,target_res)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"MOM1","mom1",this_source,this_data,line,folder,target_res)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"EMOM1","emom1",this_source,this_data,line,folder,target_res)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"MOM2","mom2",this_source,this_data,line,folder,target_res)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"EMOM2","emom2",this_source,this_data,line,folder,target_res)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"TPEAK","tpeak",this_source,this_data,line,folder,target_res)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],target_slice,"RMS","rms",this_source,this_data,line,folder,target_res)
            
