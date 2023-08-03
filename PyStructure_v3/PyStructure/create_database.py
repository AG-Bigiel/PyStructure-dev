"""
This routine generates a dictionary, similar to  the struct in idl.
Several side functions are part of this routine.
The original routine was wrirtten in idl (see create_database.pro)

The output is a dictionary saved as an .npy file. To open it in a new python
script use for example:

    > read_dictionary = np.load('datafile.npy',allow_pickle = True).item()
    > ra_samp = read_dictionary["ra_deg"]
    > dec_samp = read_dictionary["dec_deg"]
    > intensity = np.nansum(read_dictionary["SPEC_VAL_CO21"], axis = 0)
    > plt.scatter(ra_samp, dec_samp, c = intensity, marker = "h")

MODIFICATION HISTORY
    -   v1.0.1 16-22 October 2019: Conversion from IDL to Python
        Minor changes implemented
        ToDo:
            - now can only read in (z,x,y) cubes, but should be flexible to
              recognize (1,z,x,y) cubes as well

    - v1.1.1 26 October 2020: More stable version. Several bugs fixed.
            - Used by whole Bonn group
    
    - v1.2.1 January 2022
            - Implemented customization of reference line for masking.
              Now several lines can be defined for the mask creation
    
    - v1.2.2 January 2022
            - Implement Moment 1, Moment 2 and EW calculation
            - Restructured INT and SPEC keys (Mom maps now in INT keys)
            
    - v2.0.0 January 2022
            - Implemented config file: You can run the PyStructure using a single config file
    - v2.0.1. January 2022
            - Automatically determine the max radius for the sampling points
    - v2.1.0. July 2022
            - Include Spectral Smooting and Convolving for data with significantly different spectral resolution.
    - v2.1.1. October 2022
            - Save moment maps as fits file

"""
__author__ = "J. den Brok"
__version__ = "v2.1.1"
__email__ = "jdenbrok@astro.uni-bonn.de"
__credits__ = ["L. Neumann","M. Jimenez-Donaire", "E. Rosolowsky","A. Leroy ", "I. Beslic"]


import numpy as np
import pandas as pd
import os.path
from os import path
import shutil
from astropy.io import fits
from datetime import date
import argparse
today = date.today()
date_str = today.strftime("%Y_%m_%d")

import sys
sys.path.append("./scripts/")
from structure_addition import *
from sampling import *
from sampling_at_resol import *
from deproject import *
from twod_header import *
from making_axes import *
from processing_spec import *
from message_list import *
from save_moment_maps import *
#----------------------------------------------------------------------
# Change these lines of code with correct directory and names
#----------------------------------------------------------------------

# <path to directory with the data files>
data_dir = "data/"

# <filename of geometry file>
geom_file = "List_Files/geometry.txt"
# <filename of band file>
band_file = "List_Files/band_list.txt"
# <filename of cube file>
cube_file = "List_Files/cube_list.txt"
# <filename of overlay or mask> #should be stored in data_dir
overlay_file = "_12co21.fits"

# <Output Directory for Dictionaries>
out_dic = "Output/"

# Set the target resolution for all data in arcseconds (if resolution set to angular)
target_res = 27.


#!!!!!!!!!!!!!Advanced------------------------------------------
NAXIS_shuff = 200
CDELT_SHUFF = 4000.  #m/s
spacing_per_beam = 2 #default, use half beam spacing
# give number (in units deg) or set to "auto"
max_rad = "auto" #default extension of the map in deg (increase, if you map is larger)

"""
angular: use target_res in as
physical: convert target_res (in pc) to as
native: use the angular resolution of the overlay image
"""
resolution = 'angular'

# Save the convolved cubes & bands
save_fits = False

"""
Define which line to use as reference line for the spectral processing
"first": use first line in cube_list as reference line
"<LINE_NAME>": Use line name as reference line
"all": Use all lines in cube for mask
n: (integer) use first n lines as reference. n=0 is same result as "first".
"ref+HI": Use first line and HI
"""
ref_line = "first"

#define upper and lower mask threshold (S/N)
SN_processing = [2,4]

#define SN threshold for Mom1, Mom2 and EW calculation (for individual lines)
mom_thresh = 3
#differentiate between "fwhm", "sqrt", or "math"
# math: use mathematical definition
# sqrt: take square-root of mom2
# fwhm: convert sqrt(mom2) to fwhm
mom2_method = "fwhm"


"""
Spectral smoothing

"default": Do not perform any spectral smoothing
"overlay": Perform spectral smoothing to spectral resolution of overlay cube
n: float – convolve to spectral resolution n [km/s]
"""
spec_smooth = "default"

"""
define the way the spectral smoothing should be performed:
"binned": binn channels together (to nearest integer of ratio theta_target/theta_nat)
"gauss": perform convolution with gaussian kernel (theta_target^2-theta_nat^2)**0.5
!!!! Warning, gaussian smoothing seems to systematicaly underestimate the rms by 10-15%
"combined": do the binned smoothing first (to nearest integer ratio) and then the rest via Gauss
"""
spec_smooth_method = "binned"


"""
Save the created moment maps as fits file
"""
save_mom_maps = False

#folder to save fits files in
folder_savefits="./saved_FITS_files/"
#---------------------------------------------------------------


#----------------------------------------------------------------------
# The function that generates an empty directory
#----------------------------------------------------------------------
def empire_record_header():
    """
    Make the first, general fields for an EMPIRE database record.
    """

    new_structure_empire = {
    "source": '',
    "ra_deg": None,
    "dec_deg": None,
    "dist_mpc": None,
    "posang_deg": None,
    "incl_deg": None,
    "beam_as": None,
    "rgal_as": None,
    "rgal_kpc": None,
    "rgal_r25": None,
    "theta_rad": None
    }

    return new_structure_empire


def create_temps(conf_file):
    """
    Separeate the config file into variables, band and cube list
    """
    loc = 0
    py_input ='./Temp_Files/conf_Py.py'
    band_f = './Temp_Files/band_list_temp.txt'
    cube_f = './Temp_Files/cube_list_temp.txt'
    
    with open(conf_file,'r') as firstfile, open(py_input,'a') as secondfile, open(band_f,'a') as third, open(cube_f,'a') as fourth:
      
        # read content from first file
        for line in firstfile:
            # append content to second file
            if "Define Bands" in line:
                loc = 1
            if "Define Cubes" in line:
                loc = 2
        
            if loc == 0:
                secondfile.write(line)
            elif loc ==1:
                third.write(line)
            elif loc ==2:
                fourth.write(line)
                
    return band_f, cube_f

def create_database(just_source=None, quiet=False, conf=False):
    """
    Function that generates a python dictionary containing a hexagonal grid.
    :param just_source: String name of a source, if one wants only one galaxy
    :param quiet: Verbosity set to mute
    :param conf: Config File provided
    :return database: python dictionary
    """

    if quiet == False:
        print("[INFO]\t Reading in galaxy parameters.")
    names_glxy = ["galaxy", "ra_ctr", "dec_ctr", "dist_mpc", "e_dist_mpc",
                  "incl_deg", "e_incl_deg","posang_deg", "e_posang_deg",
                  "r25", "e_r25"]
    glxy_data = pd.read_csv(geom_file, sep = "\t",names = names_glxy,
                            comment = "#")
    
    #define list of sources (need to differentiate between conf file input and default)
    if conf:
        if isinstance(sources, tuple):
            galaxy_list = list(sources)
        else:
            galaxy_list = [sources]
        
    else:
        galaxy_list = list(glxy_data["galaxy"])
    
    n_sources = len(galaxy_list)
    # -----------------------------------------------------------------
    # GENERATE THE EMPTY DATA STRUCTURE
    # -----------------------------------------------------------------
    if quiet == False:
        print("[INFO]\t Generating new dictionary.")
    empty_structure = empire_record_header()

    # Add the bands to the structure
    band_columns = ["band_name","band_desc", "band_unit",
                    "band_ext", "band_dir","band_uc" ]
    bands = pd.read_csv(band_file, names = band_columns, sep='[\s,]{2,20}', comment="#")

    n_bands = len(bands["band_name"])
    for ii in range(n_bands):
        empty_structure = add_band_to_struct(struct=empty_structure,
                                         band=bands["band_name"][ii],
                                         unit=bands["band_unit"][ii],
                                         desc=bands["band_desc"][ii])

    if quiet == False:
        print("[INFO]\t {} band(s) added to structure.".format(n_bands))


    # Add the cubes to the structure
    cube_columns = ["line_name", "line_desc", "line_unit", "line_ext", "line_dir" , "band_ext", "band_uc"]

    cubes = pd.read_csv(cube_file, names = cube_columns, sep='[\s,]{2,20}', comment="#")
    n_cubes = len(cubes["line_name"])
    for ii in range(n_cubes):
        empty_structure = add_spec_to_struct(struct=empty_structure,
                                         line=cubes["line_name"][ii],
                                         unit=cubes["line_unit"][ii],
                                         desc=cubes["line_desc"][ii])
        
        # if we provide a cube for which we already have the 2D map, include it as a band
        if not cubes["band_ext"].isnull()[ii]:
            empty_structure = add_band_to_struct(struct=empty_structure,
                                                    band=cubes["line_name"][ii],
                                                    unit=cubes["line_unit"][ii]+"km/s",
                                                    desc=cubes["line_desc"][ii])

    if quiet == False:
        print("[INFO]\t {} cube(s) added to structure.".format(n_cubes))

    #-----------------------------------------------------------------
    # LOOP OVER SOURCES
    #-----------------------------------------------------------------

    #additional parameters
    run_success = [True]*n_sources #keep track if run succesfull for each galaxy
    fnames=[""]*n_sources   #filename save for galaxy
    overlay_hdr_list = []
    overlay_slice_list = []
    
    for ii in range(n_sources):
        #if config file provided, use the list of galaxies provided therein
        
        this_source = galaxy_list[ii]
        
        if not this_source in list(glxy_data["galaxy"]):
            run_success[ii]=False

            print("[ERROR]\t "+this_source+" Not in galaxy table.")

            continue
            
        #assign correct index of list and input galaxy (relevant for index file)
        ii_list = np.where(np.array(glxy_data["galaxy"])==this_source)[0][0]
        

        if not just_source is None:
            if this_source != just_source:
                continue

        print("-------------------------------")
        print("Galaxy "+this_source)
        print("-------------------------------")

    #---------------------------------------------------------------------
    # MAKE SAMPLING POINTS FOR THIS TARGET
    #--------------------------------------------------------------------

     #Generate sampling points using the overlay file provided as a template and half-beam spacing.


        # check if overlay name given with or without the source name in it:
        if this_source in overlay_file:
            overlay_fname = data_dir+overlay_file
        else:
            overlay_fname = data_dir+this_source+overlay_file


        if not path.exists(overlay_fname):
            run_success[ii]=False

            print("[ERROR]\t No Overlay data found. Skipping "+this_source+". Check path to overlay file.")
            overlay_hdr_list.append("")
            overlay_slice_list.append("")
            continue


        ov_cube,ov_hdr = fits.getdata(overlay_fname, header = True)
        
       
        #check, that cube is not 4D
        if ov_hdr["NAXIS"]==4:
            run_success[ii]=False
            overlay_hdr_list.append("")
            overlay_slice_list.append("")
            print("[ERROR]\t 4D cube provided. Need 3D overlay. Skipping "+this_source)
            continue
            
        #add slice of overlay
        overlay_slice_list.append(ov_cube[ov_hdr["NAXIS3"]//2,:,:])
        
        this_vaxis_ov = make_axes(ov_hdr, vonly = True)
        #mask = total(finite(hcn_cube),3) ge 1
        mask = np.sum(np.isfinite(ov_cube), axis = 0)>=1
        mask_hdr = twod_head(ov_hdr)
        overlay_hdr_list.append(mask_hdr)
        if resolution == 'native':
            target_res_as = np.max([ov_hdr['BMIN'], ov_hdr['BMAJ']]) * 3600
        elif resolution == 'physical':
            target_res_as = 3600 * 180/np.pi * 1e-6 * target_res / glxy_data['dist_mpc'][ii_list]
        elif resolution == 'angular':
            target_res_as = target_res
        else:
            print('[ERROR]\t Resolution keyword has to be "native","angular" or "physical".')

        # Determine
        spacing = target_res_as / 3600. / spacing_per_beam
        
        samp_ra, samp_dec = make_sampling_points(
                             ra_ctr = glxy_data["ra_ctr"][ii_list],
                             dec_ctr = glxy_data["dec_ctr"][ii_list],
                             max_rad = max_rad,
                             spacing = spacing,
                             mask = mask,
                             hdr_mask = mask_hdr,
                             overlay_in = overlay_fname,
                             show = False
                             )

        print("[INFO]\t Finshed generating Hexagonal Grid.")
    #---------------------------------------------------------------------
    # INITIIALIZE THE NEW STRUCTURE
    #--------------------------------------------------------------------
        n_pts = len(samp_ra)

        # The following lines do this_data=replicate(empty_struct, 1)

        this_data = {}
        
        #for n in range(n_pts):
        for key in empty_structure.keys():
                #this_data.setdefault(key, []).append(empty_structure[key])
                this_data[key]=empty_structure[key]
            
        # Some basic parameters for each galaxy:
        this_data["source"] = this_source
        this_data["ra_deg"] = samp_ra
        this_data["dec_deg"] = samp_dec
        this_data["dist_mpc"] = glxy_data["dist_mpc"][ii_list]
        this_data["posang_deg"] = glxy_data["posang_deg"][ii_list]
        this_data["incl_deg"] = glxy_data["incl_deg"][ii_list]
        this_data["beam_as"] = target_res_as

        # Convert to galactocentric cylindrical coordinates
        rgal_deg, theta_rad = deproject(samp_ra, samp_dec,
                                        [glxy_data["posang_deg"][ii_list],
                                         glxy_data["incl_deg"][ii_list],
                                         glxy_data["ra_ctr"][ii_list],
                                         glxy_data["dec_ctr"][ii_list]
                                        ], vector = True)


        this_data["rgal_as"] = rgal_deg * 3600
        this_data["rgal_kpc"] = np.deg2rad(rgal_deg)*this_data["dist_mpc"]*1e3
        this_data["rgal_r25"] = rgal_deg/(glxy_data["r25"][ii_list]/60.)
        this_data["theta_rad"] = theta_rad

        #---------------------------------------------------------------------
        # LOOP OVER MAPS, CONVOLVING AND SAMPLING
        #--------------------------------------------------------------------

        for jj in range(n_bands):

            this_band_file = bands["band_dir"][jj] + this_source + bands["band_ext"][jj]
            if not path.exists(this_band_file):
                print("[ERROR]\t Band "+bands["band_name"][jj] +" not found for "\
                       + this_source)

                continue

            if "/beam" in bands["band_unit"][jj]:
                perbeam = True
            else:
                perbeam = False
            this_int, this_hdr = sample_at_res(in_data=this_band_file,
                                     ra_samp = samp_ra,
                                     dec_samp = samp_dec,
                                     target_res_as = target_res_as,
                                     target_hdr = ov_hdr,
                                     show = False,
                                     line_name =bands["band_name"][jj],
                                     galaxy =this_source,
                                     path_save_fits = data_dir,
                                     save_fits = save_fits,
                                     perbeam = perbeam)


            this_tag_name = 'INT_VAL_' + bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_int
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue

#; MJ: I AM ADDING THE CORRESPONDING UNITS

            this_unit = bands["band_unit"][jj]
            this_tag_name = 'INT_UNIT_' + bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_unit
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue

#; MJ: ...AND ALSO THE UNCERTAINTIES FOR THE MAPS

            this_uc_file = bands["band_dir"][jj] + this_source + bands["band_uc"][jj]
            if not path.exists(this_uc_file):
                print("[WARNING]\t UC Band "+bands["band_name"][jj]+" not found for "+
                      this_source,)
                continue
            print('[INFO]\t Sampling at resolution band '+bands["band_name"][jj]
                   +' for '+this_source)

            this_uc, this_hdr = sample_at_res(in_data = this_uc_file,
                                    ra_samp = samp_ra,
                                    dec_samp = samp_dec,
                                    target_res_as = target_res_as,
                                    target_hdr = ov_hdr,
                                    perbeam = perbeam)
            this_tag_name = 'INT_UC_'+bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_uc
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue


            #---------------------------------------------------------------------
            # LOOP OVER MAPS, CONVOLVING AND SAMPLING
            #--------------------------------------------------------------------

        for jj in range(n_cubes):
            this_line_file = cubes["line_dir"][jj] + this_source + cubes["line_ext"][jj]


            if not path.exists(this_line_file):

                print("[ERROR]\t Line "+cubes["line_name"][jj]+" not found for "+
                      this_source)

                continue
            print('[INFO]\t Sampling at resolution band '+cubes["line_name"][jj]
                   +' for '+this_source)

            if "/beam" in cubes["line_unit"][jj]:
                perbeam = True
            else:
                perbeam = False
            this_spec, this_hdr = sample_at_res(in_data = this_line_file,
                                      ra_samp = samp_ra,
                                      dec_samp = samp_dec,
                                      target_res_as = target_res_as,
                                      target_hdr = ov_hdr,
                                      line_name =cubes["line_name"][jj],
                                      galaxy =this_source,
                                      path_save_fits = data_dir,
                                      save_fits = save_fits,
                                      perbeam = perbeam,
                                      spec_smooth = [spec_smooth,spec_smooth_method])



            this_tag_name = 'SPEC_VAL_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_spec
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue

            #this_line_hdr = fits.getheader(this_line_file)
            
            this_vaxis = make_axes(this_hdr, vonly = True)
            sz_this_spec = np.shape(this_spec)
            n_chan = sz_this_spec[1]

            for kk in range(n_pts):
                temp_spec = this_data[this_tag_name][kk]
                temp_spec[0:n_chan] = this_spec[kk,:]
                this_data[this_tag_name][kk] = temp_spec


            this_tag_name = 'SPEC_VCHAN0_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_hdr["CRVAL3"]
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue
            this_tag_name = 'SPEC_DELTAV_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_hdr["CDELT3"]
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue


            #------------------------------------------------------------------
            # Added: Check, if in addition to 3D cube, a customized 2D map is provided

            if not cubes["band_ext"].isnull()[jj]:

                this_band_file = cubes["line_dir"][jj] + this_source + cubes["band_ext"][jj]
                print("[INFO]\t For Cube "+cubes["line_name"][jj] +" a 2D map is provided.")
                if not path.exists(this_band_file):
                    print("[ERROR]\t Band "+cubes["line_name"][jj] +" not found for "\
                           + this_source)
                    print(this_band_file)

                    continue


                this_int, this_hdr = sample_at_res(in_data=this_band_file,
                                         ra_samp = samp_ra,
                                         dec_samp = samp_dec,
                                         target_res_as = target_res_as,
                                         target_hdr = ov_hdr,
                                         show = False,
                                         line_name =cubes["line_name"][jj],
                                         galaxy =this_source,
                                         path_save_fits = data_dir,
                                         save_fits = save_fits,
                                         perbeam = perbeam)


                this_tag_name = 'INT_VAL_' + cubes["line_name"][jj].upper()
                if this_tag_name in this_data:
                    this_data[this_tag_name] = this_int
                else:
                    print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                          " to the database.")
                    continue


                this_uc_file = cubes["line_dir"][jj] + this_source + str(cubes["band_uc"][jj])
                if not path.exists(this_uc_file):
                    print("[WARNING]\t UC Band "+cubes["line_name"][jj]+" not found for "+
                          this_source,)
                    continue
                print('[INFO]\t Sampling at resolution band '+cubes["line_name"][jj]
                       +' for '+this_source)

                this_uc, this_hdr = sample_at_res(in_data = this_uc_file,
                                        ra_samp = samp_ra,
                                        dec_samp = samp_dec,
                                        target_res_as = target_res_as,
                                        target_hdr = ov_hdr,
                                        perbeam = perbeam)
                this_tag_name = 'INT_UC_'+cubes["line_name"][jj].upper()
                if this_tag_name in this_data:
                    this_data[this_tag_name] = this_uc
                else:
                    print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                    continue

            print("[INFO]\t Done with line " + cubes["line_name"][jj])

        # Save the database
        if resolution == 'native':
            res_suffix = str(target_res_as).split('.')[0]+'.'+str(target_res_as).split('.')[1][0]+'as'
        elif resolution == 'angular':
            res_suffix = str(target_res_as).split('.')[0]+'as'
        elif resolution == 'physical':
            res_suffix = str(target_res).split('.')[0]+'pc'


        fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+'.npy'
        fnames[ii] = fname_dict
        np.save(fname_dict, this_data)
    #---------------------------------------------------------------------
    # NOW PROCESS THE SPECTRA
    #---------------------------------------------------------------------
    if not quiet:
        print("[INFO]\t Start processing Spectra.")
    process_spectra(glxy_data,
                    galaxy_list,
                    cubes,fnames,
                    [NAXIS_shuff, CDELT_SHUFF],
                    run_success,
                    ref_line,
                    SN_processing,
                    [mom_thresh,mom2_method],
                    )

    #Open the PyStructure and Save as FITS File
    if save_mom_maps:
        #create a folder to save
        if not os.path.exists(folder_savefits):
            os.makedirs(folder_savefits)
        # Warning
        if spacing_per_beam < 4:
            print('[WARNING]\t Spacing per beam too small for proper resampling to pixel grid.')
    
        #iterate over the individual sources
        save_mom_to_fits(fnames,
                         cubes,
                         galaxy_list,
                         run_success,
                         overlay_hdr_list,
                         overlay_slice_list,
                         folder_savefits,
                        target_res_as)
    
    return run_success

#allow input of config file
parser = argparse.ArgumentParser(description="config file")
parser.add_argument("--config")
args, leftovers = parser.parse_known_args()

#check if config file provided
config_prov = False
if not args.config is None:
    print("[INFO]\t Configure File Provided.")
    config_prov = True
    conf_file = args.config
    #if folder exists, we delete it first to make sure it contains no files
    if os.path.exists("./Temp_Files/"):
        shutil.rmtree('./Temp_Files')
    os.makedirs("./Temp_Files/")

    temp_f = create_temps(conf_file)
    band_file = temp_f[0]
    cube_file = temp_f[1]
    
    #import and use variables from config_file
    sys.path.append("./Temp_Files/")
    from conf_Py import *


run_success = create_database(conf=config_prov)

#remove the temporary folder after the run is finished
if config_prov:
    shutil.rmtree('./Temp_Files')
    
if all(run_success):
    print("[INFO]\t Run finished succesfully")
    
else:
    print("[WARNING]\t Run Terminated with potential critical error!")

#print_warning(0)
