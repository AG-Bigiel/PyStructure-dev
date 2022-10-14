"""
Author: L. Neumann + J. den Brok
"""
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import griddata
from astropy.io import fits

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
    
def save_to_fits(ra,
                 dec,
                 hdr_in,
                 key,
                 filename,
                 this_source,
                 this_data,
                 line,
                 folder="./saved_FITS_files/"):

    map_cartesian = sample_to_hdr(this_data["INT_"+key+"_"+line.upper()],
                                           ra,
                                           dec,
                                           hdr_in)
    fits.writeto(folder+this_source+"_"+line+"_"+filename+".fits", data =map_cartesian, header =  hdr_in, overwrite=True)
    
    
    

def save_mom_to_fits(fname,
                     lines_data,
                     source_list,
                     run_success,
                     target_hdr_list, folder):
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
        
        
        for line in lines_data["line_name"]:
            #iterate over the moment maps
            #mom0:
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"VAL","mom0",this_source,this_data,line,folder)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"UC","emom0",this_source,this_data,line,folder)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"MOM1","mom1",this_source,this_data,line,folder)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"EMOM1","emom1",this_source,this_data,line,folder)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"MOM2","mom2",this_source,this_data,line,folder)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"EMOM2","emom2",this_source,this_data,line,folder)
            
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"TPEAK","tpeak",this_source,this_data,line,folder)
            save_to_fits(ra_deg,dec_deg,target_hdr_list[ii],"RMS","rms",this_source,this_data,line,folder)
            
