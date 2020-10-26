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
            
    - v1.1.1 26 October 2020: More stable version. Several bugs fixed
    
"""
__author__ = "J. den Brok"
__version__ = "v1.0.1"
__email__ = "jdenbrok@astro.uni-bonn.de"
__credits__ = ["A. Leroy ", "I. Beslic"]


import numpy as np
import pandas as pd
import os.path
from os import path
from astropy.io import fits
from datetime import date
today = date.today()
date_str = today.strftime("%Y_%m_%d")

from structure_addition import *
from sampling import *
from sampling_at_resol import *
from deproject import *
from twod_header import *
from making_axes import *
from processing_spec import *

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
target_res = 23.


#!!!!!!!!!!!!!Advanced------------------------------------------
NAXIS_shuff = 200
CDELT_SHUFF = 4000.  #m/s

"""
angular: use target_res in as
physical: convert target_res (in pc) to as
natice: use the angular resolution of the overlay image
"""
resolution = 'angular'

# Save the convolved cubes & bands
save_fits = False

#---------------------------------------------------------------


#----------------------------------------------------------------------
# The function that generates an empty directory
#----------------------------------------------------------------------
def empire_record_header():
    """
    Make the first, general fields for an EMPIRE database record.
    """

    new_structure_empire = {
    "gal": '',
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




def create_database(just_source=None, quiet=False):
    """
    Function that generates a python dictionary containing a hexagonal grid.
    :param just_source: String name of a source, if one wants only one galaxy
    :param quiet: Verbosity set to mute
    :return database: python dictionary
    """

    if quiet == False:
        print("[INFO]\t Reading in galaxy parameters.")
    names_glxy = ["galaxy", "ra_ctr", "dec_ctr", "dist_mpc", "e_dist_mpc",
                  "incl_deg", "e_incl_deg","posang_deg", "e_posang_deg",
                  "r25", "e_r25"]
    glxy_data = pd.read_csv(geom_file, sep = "\t",names = names_glxy,
                            comment = "#")
    n_sources = len(glxy_data["galaxy"])
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
    fnames=[]   #filename save for galaxy
    
    for ii in range(n_sources):

        this_source = glxy_data["galaxy"][ii]

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
            
            continue


        ov_cube,ov_hdr = fits.getdata(overlay_fname, header = True)
        
        #check, that cube is not 4D
        if ov_hdr["NAXIS"]==4:
            run_success[ii]=False
            print("[ERROR]\t 4D cube provided. Need 3D overlay. Skipping "+this_source)
            continue
        this_vaxis_ov = make_axes(ov_hdr, vonly = True)
        #mask = total(finite(hcn_cube),3) ge 1
        mask = np.sum(np.isfinite(ov_cube), axis = 0)>=1
        mask_hdr = twod_head(ov_hdr)
        
        if resolution == 'native':
            target_res_as = np.max([ov_hdr['BMIN'], ov_hdr['BMAJ']]) * 3600
        elif resolution == 'physical':
            target_res_as = 3600 * 180/np.pi * 1e-6 * target_res_pc / glxy_data['dist_mpc'][ii]
        elif resolution == 'angular':
            target_res_as = target_res
        else:
            print('[ERROR]\t Resolution keyword has to be "native","angular" or "physical".')
            
        # Determine
        spacing = target_res_as / 3600. / 2.0

        samp_ra, samp_dec = make_sampling_points(
                             ra_ctr = glxy_data["ra_ctr"][ii],
                             dec_ctr = glxy_data["dec_ctr"][ii],
                             max_rad = 0.1,
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
        for n in range(n_pts):
            for key in empty_structure.keys():
                this_data.setdefault(key, []).append(empty_structure[key])


        # Some basic parameters for each galaxy:
        this_data["gal"] = this_source
        this_data["ra_deg"] = samp_ra
        this_data["dec_deg"] = samp_dec
        this_data["dist_mpc"] = glxy_data["dist_mpc"][ii]
        this_data["posang_deg"] = glxy_data["posang_deg"][ii]
        this_data["incl_deg"] = glxy_data["incl_deg"][ii]
        this_data["beam_as"] = target_res_as

        # Convert to galactocentric cylindrical coordinates
        rgal_deg, theta_rad = deproject(samp_ra, samp_dec,
                                        [glxy_data["posang_deg"][ii],
                                         glxy_data["incl_deg"][ii],
                                         glxy_data["ra_ctr"][ii],
                                         glxy_data["dec_ctr"][ii]
                                        ], vector = True)


        this_data["rgal_as"] = rgal_deg * 3600
        this_data["rgal_kpc"] = np.deg2rad(rgal_deg)*this_data["dist_mpc"]*1e3
        this_data["rgal_r25"] = rgal_deg/(glxy_data["r25"][ii]/60.)
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
            this_int = sample_at_res(in_data=this_band_file,
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

            this_uc = sample_at_res(in_data = this_uc_file,
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
            this_spec = sample_at_res(in_data = this_line_file,
                                      ra_samp = samp_ra,
                                      dec_samp = samp_dec,
                                      target_res_as = target_res_as,
                                      target_hdr = ov_hdr,
                                      line_name =cubes["line_name"][jj],
                                      galaxy =this_source,
                                      path_save_fits = data_dir,
                                      save_fits = save_fits,
                                      perbeam = perbeam)



            this_tag_name = 'SPEC_VAL_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_spec
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue

            this_line_hdr = fits.getheader(this_line_file)

            this_vaxis = make_axes(this_line_hdr, vonly = True)
            sz_this_spec = np.shape(this_spec)
            n_chan = sz_this_spec[0]

            for kk in range(n_pts):
                temp_spec = this_data[this_tag_name][kk]
                temp_spec[0:n_chan] = this_spec[kk,:]
                this_data[this_tag_name][kk] = temp_spec


            this_tag_name = 'SPEC_VCHAN0_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_vaxis_ov[0]
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue
            this_tag_name = 'SPEC_DELTAV_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_vaxis_ov[1]-this_vaxis_ov[0]
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


                this_int = sample_at_res(in_data=this_band_file,
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

                this_uc = sample_at_res(in_data = this_uc_file,
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
            res_suffix = str(target_res_pc).split('.')[0]+'pc'
            
            
        fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+'.npy'
        fnames.append(fname_dict)
        np.save(fname_dict, this_data)
    #---------------------------------------------------------------------
    # NOW PROCESS THE SPECTRA
    #---------------------------------------------------------------------
    if not quiet:
        print("[INFO]\t Start processing Spectra.")
    process_spectra(glxy_data, cubes,fnames, [NAXIS_shuff, CDELT_SHUFF],run_success)

    
    return run_success

run_success = create_database()

if all(run_success):
    print("[INFO]\t Run finished succesfully")
else:
    print("[WARNING]\t Run Terminated with potential critical error!")
