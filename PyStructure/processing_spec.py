import numpy as np
import pandas as pd
from scipy import stats
from astropy.stats import median_absolute_deviation, mad_std
from mom_computer import get_mom_maps

from structure_addition import *
from shuffle_spec import *


def construct_mask(ref_line, this_data, SN_processing):
    """
    Function to construct the mask based on high and low SN cut
    """
    ref_line_data = this_data["SPEC_VAL_"+ref_line]
    n_pts = np.shape(ref_line_data)[0]
    n_chan = np.shape(ref_line_data)[1]

    line_vaxis = this_data['SPEC_VCHAN0_'+ref_line]+np.arange(n_chan)*this_data['SPEC_DELTAV_'+ref_line]

    line_vaxis = line_vaxis/1000 #to km/s
    #Estimate rms
    rms = median_absolute_deviation(ref_line_data, axis = None, ignore_nan = True)
    rms = median_absolute_deviation(ref_line_data[np.where(ref_line_data<3*rms)], ignore_nan = True)

    # Mask each spectrum
    low_tresh, high_tresh = SN_processing[0], SN_processing[1]
    mask = np.array(ref_line_data > high_tresh * rms, dtype = int)
    low_mask = np.array(ref_line_data > low_tresh * rms, dtype = int)

    mask = mask & (np.roll(mask, 1,1) | np.roll(mask,-1,1))

    #remove spikes along spectral axis:
    mask = np.array((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1))>=3, dtype = int)
    low_mask = np.array((low_mask + np.roll(low_mask, 1, 1) + np.roll(low_mask, -1, 1))>=3, dtype = int)

    #remove spikes along spatial axis:
    #mask = np.array((mask + np.roll(mask, 1, 0) + np.roll(mask, -1, 0))>=3, dtype = int)
    #low_mask = np.array((low_mask + np.roll(mask, 1, 0) + np.roll(low_mask, -1, 0))>=3, dtype = int)

    #expand to cover all > 2sigma that have a 2-at-4sigma core
    for kk in range(5):
        mask = np.array(((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) >= 1), dtype = int)*low_mask

    #expand to cover part of edge of the emission line
    for kk in range(2):
        mask = np.array(((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) >= 1), dtype = int)

    # Derive the ref line mean velocity
    line_vmean = np.zeros(n_pts)*np.nan

    for jj in range(n_pts):
        line_vmean[jj] = np.nansum(line_vaxis * ref_line_data[jj,:]*mask[jj,:])/ \
                       np.nansum(ref_line_data[jj,:]*mask[jj,:])

    return mask, line_vmean, line_vaxis

def process_spectra(sources_data,
                    lines_data,
                    fname,shuff_axis,
                    run_success,
                    ref_line_method,
                    SN_processing,
                    mom_calc = [3, "fwhm"],
                    just_source = None
                    ):
    """
    :param sources_data: Pandas DataFrame which is the geometry.txt file
    :param lines_data:   Pandas DataFrame which is the cubes_list.txt
    """

    n_sources = len(sources_data["galaxy"])
    n_lines = len(lines_data["line_name"])
    if ref_line_method in list(lines_data["line_name"]):
        #user defined reference line
        ref_line = ref_line_method.upper()
    else:
        ref_line = lines_data["line_name"][0].upper()

    for ii in range(n_sources):


        #if the run was not succefull, don't do processing of the data
        if not run_success[ii]:
            continue

        this_source = sources_data["galaxy"][ii]
        if not just_source is None:
            if just_source != this_source:
                continue



        print("----------------------------------")
        print("Galaxy "+ this_source)
        print("----------------------------------")

        this_data = np.load(fname[ii],allow_pickle = True).item()
        tags = this_data.keys()

        #--------------------------------------------------------------
        #  Build a mask based on reference line(s)
        #--------------------------------------------------------------

        # Use function for mask
        mask, ref_line_vmean, ref_line_vaxis = construct_mask(ref_line, this_data, SN_processing)
        this_data["SPEC_MASK_"+ref_line]= mask
        this_data["INT_VAL_V"+ref_line] = ref_line_vmean

        #check if all lines used as reference line
        n_mask = 0
        if ref_line_method in ["all"]:
            n_mask = n_lines
            print("[INFO]\tAll lines used as prior")
        elif isinstance(ref_line_method, int):
            n_mask = np.min([n_lines,ref_line_method])
            print("[INFO]\tUsing first "+str(n_mask+1)+" lines as prior")
        if n_mask>0:
            for n_mask_i in range(1,n_mask):
                line_i = lines_data["line_name"][n_mask_i].upper()
                mask_i, ref_line_vmean_i, ref_line_vaxis_i = construct_mask(line_i, this_data, SN_processing)
                this_data["SPEC_MASK_"+line_i]= mask_i
                this_data["INT_VAL_V"+line_i] = ref_line_vmean_i

                # add mask to existing mask
                mask = mask | mask_i
        #store the mask in the PyStructure
        this_data["SPEC_MASK"]= mask


        #-------------------------------------------------------------------
        # Apply the CO-based mask to the EMPIRE lines and shuffle them
        #-------------------------------------------------------------------
        n_chan_new = 200

        for jj in range(n_lines):
            line_name = lines_data["line_name"][jj].upper()

            # need to add band structure, if the 2D was not yet provided
            if lines_data["band_ext"].isnull()[jj]:
                this_data = add_band_to_struct(struct = this_data, \
                                           band = line_name,\
                                           unit = 'K km/s', \
                                           desc = line_name + ' Shuffled by '+ref_line)

            this_data = add_spec_to_struct(struct= this_data, \
                                           line = "SHUFF"+line_name,\
                                           unit = "K",\
                                           desc = line_name + ' Shuffled by '+ref_line,\
                                           n_chan = n_chan_new)


            if not 'SPEC_VAL_'+line_name in this_data.keys():
                print("[ERROR]\t Tag for line "+line_name+ "not found. Proceeding.")
                continue
            this_spec = this_data['SPEC_VAL_'+line_name]
            if np.nansum(this_spec, axis = None)==0:
                print("[ERROR]\t Line "+line_name+" appears empty. Skipping")
                continue

            dim_sz = np.shape(this_spec)
            n_pts = dim_sz[0]
            n_chan = dim_sz[1]
            this_v0 = this_data["SPEC_VCHAN0_"+line_name]
            this_deltav = this_data["SPEC_DELTAV_"+line_name]

            this_vaxis = (this_v0 + np.arange(n_chan)*this_deltav)/1000 #to km/s


            shuffled_mask = shuffle(spec = mask, \
                                    vaxis = ref_line_vaxis,\
                                    zero = 0.0,\
                                    new_vaxis = this_vaxis, \
                                    interp = 0)
                        
            #compute moment_maps
            mom_maps = get_mom_maps(this_spec, shuffled_mask,this_vaxis, mom_calc)

            # Save in structure
            if lines_data["band_ext"].isnull()[jj]:

                tag_ii = "INT_VAL_"+line_name
                tag_uc = "INT_UC_" + line_name
                
                tag_tpeak = "INT_TPEAK_" + line_name
                tag_rms = "INT_RMS_" + line_name
                
                tag_mom1 = "INT_MOM1_" + line_name
                tag_mom1_err = "INT_EMOM1_" + line_name
                
                #Note that Mom2 corresponds to a FWHM
                tag_mom2 = "INT_MOM2_" + line_name
                tag_mom2_err = "INT_EMOM2_" + line_name
                
                tag_ew = "INT_EW_" + line_name
                tag_ew_err = "INT_EEW_" + line_name
                
                # store the different calculations
                this_data[tag_ii] = mom_maps["mom0"]
                this_data[tag_uc] = mom_maps["mom0_err"]
                this_data[tag_tpeak] = mom_maps["tpeak"]
                this_data[tag_rms] = mom_maps["rms"]
                this_data[tag_mom1] = mom_maps["mom1"]
                this_data[tag_mom1_err] = mom_maps["mom1_err"]
                this_data[tag_mom2] = mom_maps["mom2"]
                this_data[tag_mom2_err] = mom_maps["mom2_err"]
                
                this_data[tag_ew] = mom_maps["ew"]
                this_data[tag_ew_err] = mom_maps["ew_err"]
                
                #-------------------------------------------------
                #!!!!!!!! Will be depricated in future update!!!!!
                tag_tpeak_dep = "SPEC_TPEAK_" + line_name
                tag_rms_dep = "SPEC_RMS_" + line_name
                this_data[tag_tpeak_dep] = mom_maps["tpeak"]
                this_data[tag_rms_dep] = mom_maps["rms"]
                #-------------------------------------------------
            else:
                print("[INFO]\t Intensity Map for "+lines_data["line_name"][jj]+"already provided, skipping." )

            #Shuffle the line
            #;- DC modify 02 march 2017: define a reference velocity axis
            #;-   this_deltav varies from dataset to dataset (fixing bug for inverted CO21 vaxis)
            cdelt = shuff_axis[1]
            naxis_shuff = int(shuff_axis[0])
            new_vaxis = cdelt * (np.arange(naxis_shuff)-naxis_shuff/2)
            new_vaxis=new_vaxis/1000 #to km/s

            shuffled_line = shuffle(spec = this_spec,\
                                    vaxis = this_vaxis,\
                                    zero = ref_line_vmean,
                                    new_vaxis = new_vaxis,\
                                    interp = 0)

            tag_i = "SPEC_VAL_SHUFF" + line_name
            tag_v0 = "SPEC_VCHAN0_SHUFF" + line_name
            tag_deltav = "SPEC-DELTAV_SHUFF" + line_name


            this_data[tag_i] = shuffled_line
            this_data[tag_v0] = new_vaxis[0]
            this_data[tag_deltav] = (new_vaxis[1] - new_vaxis[0])


        np.save(fname[ii], this_data)


        # /__
