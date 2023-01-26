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
from message_list import *
from shuffle_spec import *
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




def create_database(source_info,band_file,cube_file, quiet=False):
    """
    Function that generates a python dictionary containing a hexagonal grid.
    :param just_source: String name of a source, if one wants only one galaxy
    :param quiet: Verbosity set to mute
    :param conf: Config File provided
    :return database: python dictionary
    """

    sources = source_info[0]
    ra_ctr = source_info[1]
    dec_ctr = source_info[2]
    posang_deg =source_info[3]
    incl_deg =source_info[4]
    dist_mpc =source_info[5]
    target_res_as = source_info[6]
    max_rad = source_info[7]
    data_dir =source_info[8]
    spacing_per_beam =source_info[9]
    vmap_file = source_info[10]
    out_dic = source_info[11]
    shuff_axis =source_info[12]
    
    if isinstance(sources, tuple):
        source_list = list(sources)
    else:
        source_list = [sources]
        
    n_sources = len(source_list)
    # -----------------------------------------------------------------
    # GENERATE THE EMPTY DATA STRUCTURE
    # -----------------------------------------------------------------
    
    if quiet == False:
        print("[INFO]\t Generating new dictionary.")
    empty_structure = empire_record_header()

    # Add the bands to the structure
    band_columns = ["band_name", "band_ext", "band_dir"]
    bands = pd.read_csv(band_file, names = band_columns, sep='[\s,]{2,20}', comment="#")

    n_bands = len(bands["band_name"])
    for ii in range(n_bands):
        empty_structure = add_band_to_struct(struct=empty_structure,
                                         band=bands["band_name"][ii],
                                         unit="",
                                         desc="")

    if quiet == False:
        print("[INFO]\t {} band(s) added to structure.".format(n_bands))


    # Add the cubes to the structure
    cube_columns = ["line_name",  "line_ext", "line_dir"]

    cubes = pd.read_csv(cube_file, names = cube_columns, sep='[\s,]{2,20}', comment="#")
    n_cubes = len(cubes["line_name"])
    for ii in range(n_cubes):
        empty_structure = add_spec_to_struct(struct=empty_structure,
                                         line=cubes["line_name"][ii],
                                         unit="",
                                         desc="")

        
    if quiet == False:
        print("[INFO]\t {} cube(s) added to structure.".format(n_cubes))

    #-----------------------------------------------------------------
    # LOOP OVER SOURCES
    #-----------------------------------------------------------------

    #additional parameters
    fnames=[""]*n_sources   #filename save for galaxy
    overlay_hdr_list = []
    overlay_slice_list = []
    
    for ii in range(n_sources):
        #if config file provided, use the list of galaxies provided therein
        
        this_source = source_list[ii]
        

        print("-------------------------------")
        print("Galaxy "+this_source)
        print("-------------------------------")

    #---------------------------------------------------------------------
    # MAKE SAMPLING POINTS FOR THIS TARGET
    #--------------------------------------------------------------------

     #Generate sampling points using the overlay file provided as a template and half-beam spacing.


        # check if overlay name given with or without the source name in it:
        overlay_fname = data_dir+this_source+cubes["line_ext"][0]

        ov_cube,ov_hdr = fits.getdata(overlay_fname, header = True)
        
       
        #check, that cube is not 4D
        if ov_hdr["NAXIS"]==4:
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
        
        
        # Determine
        spacing = target_res_as / 3600. / spacing_per_beam
        
        samp_ra, samp_dec = make_sampling_points(
                             ra_ctr = ra_ctr,
                             dec_ctr = dec_ctr,
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
        for n in range(n_pts):
            for key in empty_structure.keys():
                this_data.setdefault(key, []).append(empty_structure[key])


        # Some basic parameters for each galaxy:
        this_data["gal"] = this_source
        this_data["ra_deg"] = samp_ra
        this_data["dec_deg"] = samp_dec

        # Convert to galactocentric cylindrical coordinates
        rgal_deg, theta_rad = deproject(samp_ra, samp_dec,
                                        [posang_deg,
                                         incl_deg,
                                         ra_ctr,
                                         dec_ctr
                                        ], vector = True)


        this_data["rgal_as"] = rgal_deg * 3600
        this_data["rgal_kpc"] = np.deg2rad(rgal_deg)*dist_mpc*1e3
        this_data["theta_rad"] = theta_rad

        #---------------------------------------------------------------------
        # LOOP OVER MAPS, CONVOLVING AND SAMPLING
        #--------------------------------------------------------------------
        #add the velocity map
        
        this_int, this_hdr = sample_at_res(in_data=vmap_file,
                                     ra_samp = samp_ra,
                                     dec_samp = samp_dec,
                                     target_res_as = target_res_as,
                                     target_hdr = ov_hdr,
                                     show = False,
                                     line_name ="VMAP",
                                     galaxy =this_source,
                                     )


        this_tag_name = 'INT_VAL_VMAP'
        this_data[this_tag_name] = this_int
        
        for jj in range(n_bands):

            this_band_file = bands["band_dir"][jj] + this_source + bands["band_ext"][jj]
            if not path.exists(this_band_file):
                print("[ERROR]\t Band "+bands["band_name"][jj] +" not found for "\
                       + this_source)

                continue

            
            this_int, this_hdr = sample_at_res(in_data=this_band_file,
                                     ra_samp = samp_ra,
                                     dec_samp = samp_dec,
                                     target_res_as = target_res_as,
                                     target_hdr = ov_hdr,
                                     show = False,
                                     line_name =bands["band_name"][jj],
                                     galaxy =this_source,
                                     )


            this_tag_name = 'INT_VAL_' + bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_int
            else:
                print("[ERROR]\t  I had trouble matching tag "+this_tag_name+
                      " to the database.")
                continue

#; MJ: I AM ADDING THE CORRESPONDING UNITS

            


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

            this_spec, this_hdr = sample_at_res(in_data = this_line_file,
                                      ra_samp = samp_ra,
                                      dec_samp = samp_dec,
                                      target_res_as = target_res_as,
                                      target_hdr = ov_hdr,
                                      line_name =cubes["line_name"][jj],
                                      galaxy =this_source,
                                     )



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


            cdelt = shuff_axis[1]
            naxis_shuff = int(shuff_axis[0])
            new_vaxis = cdelt * (np.arange(naxis_shuff)-naxis_shuff/2)
            new_vaxis=new_vaxis/1000 #to km/s

            
            shuffled_line = shuffle(spec = this_data['SPEC_VAL_'+cubes["line_name"][jj].upper()],\
                                    vaxis = this_vaxis/1000,\
                                    zero = this_data["INT_VAL_VMAP"],
                                    new_vaxis = new_vaxis,\
                                    interp = 0)

            tag_i = "SPEC_VAL_SHUFF" + cubes["line_name"][jj].upper()
            tag_v0 = "SPEC_VCHAN0_SHUFF" + cubes["line_name"][jj].upper()
            tag_deltav = "SPEC-DELTAV_SHUFF" + cubes["line_name"][jj].upper()


            this_data[tag_i] = shuffled_line
            this_data[tag_v0] = new_vaxis[0]
            this_data[tag_deltav] = (new_vaxis[1] - new_vaxis[0])
            this_data["new_vaxis"] = new_vaxis
            this_data["this_vaxis"] = this_vaxis
            print("[INFO]\t Done with line " + cubes["line_name"][jj])

        # Save the database
        res_suffix = str(target_res_as).split('.')[0]+'as'
        
        fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+'.npy'
        fnames[ii] = fname_dict
        np.save(fname_dict, this_data)
    #---------------------------------------------------------------------
    # NOW PROCESS THE SPECTRA
    #---------------------------------------------------------------------
    if not quiet:
        print("[INFO]\t Start processing Spectra.")

    return fnames



