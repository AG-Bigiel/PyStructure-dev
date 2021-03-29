import numpy as np
import pandas as pd
from scipy import stats
from astropy.stats import median_absolute_deviation, mad_std

from structure_addition import *
from shuffle_spec import *

def process_spectra(sources_data, lines_data,fname,shuff_axis, run_success,just_source = None):
    """
    :param sources_data: Pandas DataFrame which is the geometry.txt file
    :param lines_data:   Pandas DataFrame which is the cubes_list.txt
    """
    ref_line = lines_data["line_name"][0].upper()
    n_sources = len(sources_data["galaxy"])
    n_lines = len(lines_data["line_name"])

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
        #  Build a CO-based mask
        #--------------------------------------------------------------

        # Extract CO data
        co21 = this_data["SPEC_VAL_"+ref_line]
        n_pts = np.shape(co21)[0]
        n_chan = np.shape(co21)[1]
        n_chan_new = 200
        co_vaxis = this_data['SPEC_VCHAN0_'+ref_line]+np.arange(n_chan)*this_data['SPEC_DELTAV_'+ref_line]

        co_vaxis = co_vaxis/1000 #to km/s
        #Estimate rms
        rms = median_absolute_deviation(co21, axis = None, ignore_nan = True)
        rms = median_absolute_deviation(co21[np.where(co21<3*rms)], ignore_nan = True)

        # Mask each spectrum
        mask = np.array(co21 > 4 * rms, dtype = int)
        low_mask = np.array(co21 > 2 * rms, dtype = int)

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



        # Derive the mean CO velocity
        co_vmean = np.zeros(n_pts)*np.nan

        for jj in range(n_pts):
            co_vmean[jj] = np.nansum(co_vaxis * co21[jj,:]*mask[jj,:])/ \
                           np.nansum(co21[jj,:]*mask[jj,:])

        # Derive the mean CO intensity
        co_ii = np.nansum(co21*mask, axis = 1)*abs(co_vaxis[1]-co_vaxis[0])

        # DC 28 feb 2017, added max to have rms as UNC even if no channe;s in mask
        co_uc = max([1, max(np.sqrt(np.nansum(mask, axis = 1)))])*rms*abs(co_vaxis[1]-co_vaxis[0])

        # Shuffle the CO intensity
        # TBD

        # Save all in a structure
        this_data = add_band_to_struct(struct = this_data, \
                                       band = ref_line, \
                                       unit = 'K km/s', \
                                       desc = 'CO 2-1 Integrated Intensity')
        this_data["INT_VAL_"+ref_line] = co_ii
        this_data["INT_VAL_V"+ref_line] = co_vmean

        #-------------------------------------------------------------------
        # Apply the CO-based mask to the EMPIRE lines and shuffle them
        #-------------------------------------------------------------------

        for jj in range(n_lines):
            line_name = lines_data["line_name"][jj].upper()

            # need to add band structure, if the 2D was not yet provided
            if lines_data["band_ext"].isnull()[jj]:
                this_data = add_band_to_struct(struct = this_data, \
                                           band = line_name,\
                                           unit = 'K km/s', \
                                           desc = line_name + ' Shuffled by CO 2-1')

            this_data = add_spec_to_struct(struct= this_data, \
                                           line = "SHUFF"+line_name,\
                                           unit = "K",\
                                           desc = line_name + ' Shuffled by CO 2-1',\
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
                                    vaxis = co_vaxis,\
                                    zero = 0.0,\
                                    new_vaxis = this_vaxis, \
                                    interp = 0)

            #Derive the mean Intensity
            this_rms = np.zeros(n_pts)*np.nan

            for m in range(n_pts):
                if np.nansum(this_spec[m,:]!=0, axis = None)>=1:

                    this_rms[m] = mad_std(\
                    this_spec[m,:][np.where(np.logical_and(\
                    shuffled_mask[m,:]==0, this_spec[m,:]!=0))], ignore_nan=True)

            this_ii = np.nansum(this_spec*shuffled_mask, axis = 1)*\
                      abs(this_vaxis[1] - this_vaxis[0])

            this_uc = np.maximum(np.ones(len(np.nansum(shuffled_mask, axis = 1))),\
                      (np.sqrt(np.nansum(shuffled_mask, axis = 1))))*this_rms*\
                      abs(this_vaxis[1] - this_vaxis[0])

            this_tpeak = np.nanmax(this_spec*shuffled_mask, axis = 1)

            # Save in structure

            if lines_data["band_ext"].isnull()[jj]:

                tag_ii = "INT_VAL_"+line_name
                tag_uc = "INT_UC_" + line_name
                tag_tpeak = "SPEC_TPEAK_" + line_name
                tag_rms = "SPEC_RMS_" + line_name
                this_data[tag_ii] = this_ii
                this_data[tag_uc] = this_uc
                this_data[tag_tpeak] = this_tpeak
                this_data[tag_rms] = this_rms
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
                                    zero = co_vmean,
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
