import numpy as np
import idlsave
import pandas as pd
import stacking_func as func
import stack_specs as stck_spc
import fitting_routine as fit
import make_bin as make_bin
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation, mad_std
import copy
import os
import warnings
warnings.filterwarnings("ignore")





def get_stack(fnames, prior_lines, lines, dir_save, dir_data ='./../../data/Database/', 
              show = False, do_smooth = False, xtype = None, 
              bin_scaling = "linear", nbins = None, xmin=None, xmax=None,
              sn_limits = [2,4], no_detec_wdw = 30, pad_v = 100, line_wdw=0, 
              ignore_empties=False, weights_type=None, rms_type=None, trim_stackspec = True):
    """
    Function converted from IDL to python
    :param fnames: String of name of the PyStructure names, e.g. "ngc5194_datbase.npy"
    :param dir_save: String of directory where to save the output
    :param dir_data: String of path to the directory, where the IDL database structure is saved
    :param show: show plots [not yet fully implemented]
    :param do_smooth: [variable not yet fully implemented]
    :param xtype: string name of the quantity by which to stack, must be included in PyStructure.
    :param bin_scaling: "linear" or "log"
    :param nbins: number of bins (integer)
    :param xmin: bin minimum (set as maximum of data range if None)
    :param xmax: bin maximum (set as minimum of data range if None)
    :param sn_limits: S/N thresholds for lower and upper mask
    :param no_detec_wdw: window size over which to integrate in case no detection is found. In km/s
    :param pad_v: in km/s range at either edges to exclude from integrating or finding the mask
    :param line_wdw: window size where emission is expected to exlude from finding the mask. In km/s
    :param ignore_empties:
    :param trim_stackspec: Boolean, if true, trim the stacked spectrum to only include channels, where the overlap of all spectra is given
    :param weights_type: string name of the quantity by which to weight the stacking
    :param rms_type: can be 'iterative'
    """

    # --------------------------------------------------------------------
    # STACKING METHOD
    # --------------------------------------------------------------------
    print("[INFO]\t Stacking by "+xtype)
    

    # -----------------------------------------------------------------------
    # LOOP THROUGH THE DIFFERENT FILES/SOURCES
    # -----------------------------------------------------------------------

    if hasattr(fnames, '__len__') == False or isinstance(fnames,str):
        fnames=[fnames]

    for i in range(len(fnames)):
    
        # -----------------------------------------------------------------------
        # RESTORE THE DATA STRUCTURES
        # -----------------------------------------------------------------------
        # Measurements
        source = fnames[i]
        dir = dir_data
        file = dir + fnames[i]
        file_ext = os.path.splitext(fnames[i])[1]
        name = os.path.splitext(fnames[i])[0]
        
        is_IDL = False
        if ".idl" in file_ext:
            struct = idlsave.read(file, verbose=False, python_dict=True)
            this_data = struct["this_data"]
            is_IDL = True
        elif ".npy" in file_ext:
            this_data = np.load(file,allow_pickle = True).item()
        else:
            raise TypeError("Input Database must me of format .idl or .npy")

        # determine xtype for this galaxy -> x axis !!!!!!!!!!!!!!!!!
        xvec = this_data[xtype]

        ra = this_data["ra_deg"]
        decl = this_data["dec_deg"]
        
        #-----------------------------------------------------------------------
        # Define the weights
        #-----------------------------------------------------------------------
        weights=None
        if weights_type in ["snr_squared"]:
            print("[INFO]\t Weighting by SNR^2 with "+prior_lines[0])
            weights=(this_data["INT_VAL_"+prior_lines[0]]/this_data["INT_UC_"+prior_lines[0]])
            weights[weights<0]=0
            weights = weights**2
        elif weights_type:
            print("[INFO]\t Weighting by " + weights_type)
            weights = this_data[weights_type]
            
        #-----------------------------------------------------------------------
        # Compute bin-size the weights
        #-----------------------------------------------------------------------
        nbins, xmin_bin, xmax_bin,xmid_bin = make_bin.get_bins(xvec, bin_scaling, nbins, xmin_in=xmin, xmax_in=xmax)
        
        
        # -----------------------------------------------------------------------
        # AVERAGE PIXEL MEASUREMENTS
        # -----------------------------------------------------------------------

        #collect all pixels with xval value
        stack = func.stack_pix_by_x(this_data,lines, xtype, xvec , xmin_bin, xmax_bin,xmid_bin, median  = 'mean' ) #0= mean, 1=median

        #--------------------------------------------------------------------------
        # STACK THE SPECTRA
        # -------------------------------------------------------------------------

        # stack spectra in those pixels
        shuffled_specs = {}

        #length of the shuffeled spectrum
        nvaxis = len(this_data["SPEC_VAL_SHUFF"+prior_lines[0]][0])
        for line in lines+prior_lines:
            shuffled_specs["SPEC_VAL_SHUFF"+line] = this_data["SPEC_VAL_SHUFF"+line]


        # create a mask
        spec_prior = shuffled_specs["SPEC_VAL_SHUFF"+prior_lines[0]]


        idnans_prior = np.ones(len(spec_prior))

        for i in range(len(idnans_prior)):
            if not sum(np.array(np.isnan(spec_prior[i]), dtype = int)) == nvaxis:
                        idnans_prior[i] = 0

        comb_mask = np.array(idnans_prior, dtype=int)

        for i in range(len(comb_mask)):
            if comb_mask[i] == 1:
                spec_prior[i] = np.zeros(nvaxis)*np.nan

        shuffled_specs[prior_lines[0]+"_spec_K"]  = spec_prior


        if do_smooth == False:
            if is_IDL:
                vaxis = this_data[0]["SPEC_VCHAN0_SHUFF"+prior_lines[0]] + this_data[0]["SPEC_DELTAV_SHUFF"+prior_lines[0]] * np.arange(len(this_data[0]["SPEC_VAL_SHUFF"+prior_lines[0]]))
            else:
                vaxis = this_data["SPEC_VCHAN0_SHUFF"+prior_lines[0]] + this_data["SPEC-DELTAV_SHUFF"+prior_lines[0]] * np.arange(len(this_data["SPEC_VAL_SHUFF"+prior_lines[0]][0]))
            prior_spec = shuffled_specs["SPEC_VAL_SHUFF"+prior_lines[0]]
            #the non-shuffled spectra
            prior_spec_orig = this_data["SPEC_VAL_"+prior_lines[0]]

            stack_spec = stck_spc.stack_spec(prior_spec, xvec,xtype, nbins, xmin_bin, xmax_bin, xmid_bin, weights = weights, ignore_empties=ignore_empties, trim_stackspec=trim_stackspec, spec_orig = prior_spec_orig)
            stack[prior_lines[0]+"_spec_K"] = stack_spec["spec"]
            
            
            #save ncounts of prior line
            stack["counts_"+prior_lines[0]] =stack_spec["counts"]
            stack["ncounts_"+prior_lines[0]] =np.nanmax(stack_spec["counts"], axis=0)
            # number of spectra, where the prior has been detected and the spectrum was shuffled
            stack["ncounts_total_"+prior_lines[0]] = stack_spec["counts_total_spec"]
            stack["nbins"] = stack_spec["counts_total"]
            stack["narea_kpc2"] = 37.575*(1/3600*stack["dist_mpc"]*np.pi/180)**2* \
                              np.cos(np.pi/180*stack["incl_deg"])*stack["ncounts_total_"+prior_lines[0]]
            
            # Iterate over the different lines that need to be stacked
            for line in lines:
                spec = shuffled_specs["SPEC_VAL_SHUFF"+line]
                #the non-shuffled spectra
                spec_orig = this_data["SPEC_VAL_"+line]
                stack_spec = stck_spc.stack_spec(spec, xvec,xtype,  nbins, xmin_bin, xmax_bin, xmid_bin,weights = weights, ignore_empties=ignore_empties, trim_stackspec=trim_stackspec, spec_orig = spec_orig)
                stack[line+"_spec_K"] = stack_spec["spec"]
                stack["counts_"+line] =stack_spec["counts"]
                stack["ncounts_"+line] =np.nanmax(stack_spec["counts"], axis=0)
                stack["ncounts_total_"+line] = stack_spec["counts_total_spec"]


        # save some more galaxy parameters
        if is_IDL:
            stack["dist_mpc"] = this_data["dist_mpc"][0]
            stack["posang_deg"] = this_data["posang_deg"][0]
            stack["incl_deg"] = this_data["incl_deg"][0]
            stack["beam_as"] = this_data["beam_as"][0]
        else:
            stack["dist_mpc"] = this_data["dist_mpc"]
            stack["posang_deg"] = this_data["posang_deg"]
            stack["incl_deg"] = this_data["incl_deg"]
            stack["beam_as"] = this_data["beam_as"]

        stack["vaxis_kms"] = vaxis

        



        #--------------------------------------------------------------------------
        # Fit Gaussian to spectral line
        # -------------------------------------------------------------------------

        # Prepare the output of the fittinbf
        n_stacks = len(stack['xmid'])
        stack["prior_mask"] = np.zeros((nvaxis,n_stacks))*np.nan

        for line in lines+prior_lines:
            stack["rms_K_"+line]       = np.zeros(n_stacks)*np.nan
            stack["peak_K_"+line]      = np.zeros(n_stacks)*np.nan
            stack["ii_K_kms_"+line]    = np.zeros(n_stacks)*np.nan
            stack["uc_ii_K_kms_"+line] = np.zeros(n_stacks)*np.nan
            stack["upplim_K_kms_"+line] = np.zeros(n_stacks)*np.nan
            stack["lowlim_K_kms_"+line] = np.zeros(n_stacks)*np.nan
            stack["SNR_"+line] = np.zeros(n_stacks)*np.nan


        for j in range(len(stack['xmid'])):

           #--------------------------------------------------------------------------
           # Need to first find the bounds using a prior (the brightest line)
           # -------------------------------------------------------------------------
            v = stack["vaxis_kms"]



            # The user can give several priors. Iterate over the priors and check which one shows the stongest detection. Use that line as prior for that given bin.
            prior_specs = []
            masks = []
            rms_values = []

            sn_low = sn_limits[0]
            sn_up = sn_limits[1]
            for prior_line in prior_lines:

                prior = copy.copy(stack[prior_line+"_spec_K"][:,j])
                
                # Skip bin if empty
                if np.nansum(prior) == 0:
                    continue

                #Ignore the boundaries
                min_v = np.nanmin(vaxis[np.where(np.isfinite(prior))])
                max_v = np.nanmax(vaxis[np.where(np.isfinite(prior))])
                always_empty = np.isfinite(prior) & (np.array(vaxis < np.min([(min_v + pad_v),max_v]), dtype = int) | \
                                  np.array(vaxis > np.max([(max_v - pad_v),0]), dtype = int))
                always_spec = always_empty | np.array(np.isfinite(prior)==0)
                prior[np.where(always_spec)]=np.nan
            
                #Ignore line centre
                prior_wo_line = np.copy(prior)
                prior_wo_line[(vaxis>-line_wdw/2) & (vaxis<line_wdw/2)] = np.nan

                #Estimate rms
                if rms_type == 'iterative':
                    rms_iter = 0  # counter for iterations
                    rms_iter_max = 100  # maximum number of iterations
                    rms_old = median_absolute_deviation(prior_wo_line, axis = None,ignore_nan=True)
                    rms = median_absolute_deviation(prior_wo_line[np.where(prior_wo_line<3*rms_old)],ignore_nan=True)
                    while abs((rms-rms_old)/rms) > 0.1:
                        rms_old = rms
                        rms = np.nanstd(prior_wo_line[np.where(prior_wo_line<3*rms_old)])
                        rms_iter += 1
                        if len(prior_wo_line[np.where(prior_wo_line<3*rms)]) == 0:
                            # if above leads to an empty spectrum, there are probably no emission-free channels
                            # in this case, rms cannot be determined and we integrate over the full bandwidth to obtain a lower limit
                            print('[WARNING]\tIn bin',j+1,'/',len(stack['xmid']),'with xmid =',stack['xmid'][j], ': No line-free channels available to compute the rms.'
                                  '\n\t\tIntegrate over full bandwidth and return lower limit.')
                            rms = np.nan
                            break
                        elif rms_iter == rms_iter_max:
                            print('[WARNING]\tIn bin',j+1,'/',len(stack['xmid']),'with xmid =',stack['xmid'][j], ': Maximum number of iterations (%i) reached in computing the rms.' % rms_iter_max)
                            break
                
                elif not rms_type:
                    rms = median_absolute_deviation(prior_wo_line, axis = None,ignore_nan=True)
                    rms = median_absolute_deviation(prior_wo_line[np.where(prior_wo_line<3*rms)],ignore_nan=True)
                else:
                    print('[ERROR]\t rms_type not specified. Must be None or "iterative".')


                # Mask each spectrum
                mask = np.array(prior > sn_up * rms, dtype = int)
                low_mask = np.array(prior > sn_low * rms, dtype = int)

                mask = mask & (np.roll(mask, 1) | np.roll(mask,-1))
                #remove spikes
                mask = np.array((mask + np.roll(mask, 1) + np.roll(mask, -1))>=3, dtype = int)

                #expand to cover all > 2sigma that have a 2-at-4sigma core
                for kk in range(10):
                    mask = np.array(((mask + np.roll(mask, 1) + np.roll(mask, -1)) >= 1), dtype = int)*low_mask

                prior_specs.append(stack[prior_line+"_spec_K"][:,j])
                masks.append(mask)
                rms_values.append(rms)

                
            # go to next bin if empty
            if len(prior_specs) == 0:
                print('[WARNING]\tIn bin',j+1,'/',len(stack['xmid']),'with xmid =',stack['xmid'][j], ': No data found. Return nan values throughout for this bin.')
                continue
                
            mask = masks[0]
            rms = rms_values[0]
            prior_ii_ref = np.nansum(prior_specs[0]*mask)*abs(v[1]-v[0])
            for n in range(len(masks)-1):
                if len(masks[n+1][0])<1:
                    continue
                prior_ii_comp = np.nansum(prior_specs[n+1]*mask[n+1])*abs(v[1]-v[0])
                if prior_ii_comp>prior_ii_ref:
                    mask = masks[n+1]
                    rms = rms_values[n+1]

            stack["prior_mask"][:,j] = mask


            for line in lines+prior_lines:

                # Fit the line
                spec_to_integrate = stack[line+"_spec_K"][:,j]
                spec_to_integrate[spec_to_integrate == 0] = np.nan

                #calculate rms of line
                if np.isnan(rms):
                    # return lower limit for line intensity if mask could not be determined
                    rms_line = np.nan
                    line_ii = np.nan
                    line_uc = np.nan
                    line_lowlim = np.nansum(spec_to_integrate)*abs(v[1]-v[0])
                    stack["lowlim_K_kms_"+line][j] = line_lowlim
                else:
                    # rms of each line is the standard deviation outside of the mask
                    rms_line = np.nanstd(spec_to_integrate[np.where(mask==0)])
                    
                    # rescale rms for # spectra contributing to the stack vs # total spectra inside bin
                    if not ignore_empties:
                        rms_line *= np.sqrt(stack["ncounts_total_"+line][j]/stack["ncounts_"+line][j])
                    
                    # line intensity and uncertainty
                    line_ii = np.nansum(spec_to_integrate*mask)*abs(v[1]-v[0])
                    line_uc = max([1, np.sqrt(np.nansum(mask))])*rms_line*abs(v[1]-v[0])
                    
                    


                # Fill in dictionary
                stack["rms_K_"+line][j] = rms_line
                stack["peak_K_"+line][j] = np.nanmax(spec_to_integrate*mask)

                stack["ii_K_kms_"+line][j] = line_ii
                stack["uc_ii_K_kms_"+line][j] = line_uc

                SNR_line = line_ii/line_uc
                stack["SNR_"+line][j] = SNR_line
                          
                if SNR_line<3:    
                    # if window found: integrate over that
                    if np.nansum(mask)>2:
                        stack["upplim_K_kms_"+line][j] = 3*line_uc

                    # if no window found: use (default) 30 km/s window
                    else:
                        mask_no_detec_wdw = np.zeros_like(mask)
                        mask_no_detec_wdw[(vaxis>-no_detec_wdw/2) & (vaxis<no_detec_wdw/2)] = 1
                        stack["upplim_K_kms_"+line][j] = 3*rms_line*max([1, np.sqrt(np.nansum(mask_no_detec_wdw))])*abs(vaxis[1]-vaxis[0])
                          

        path_save = dir_save + name+"_stack_"+xtype+".npy"
        np.save(path_save, stack)
        print("[INFO]\t Successfull Finished")

    return None




"""
# Specify the directories:
final_direc = "./Example/Example_Results/"
data_direc = "./../PyStructure/Output/"
#data_direc = "./../../PhD_prep/M101_proposal/"

# As an example, we will use the databse of galaxy NGC 4321. We have only one galaxy.
# If you have more, you can expand the list accordingly
galaxies = ["ngc5457"]
galaxies = ["NGC6946"]

# We will only stack by radius for now.
# If you want more, you can expand the list accordingly
xtypes = ["12co21"]

#Define a set of lines to stack.
lines = ["CII"]
prior = ["12CO21"]

# Loop over the stacking quantities
for xtype in xtypes:
    for galaxy in galaxies:
        res = get_stack(galaxy,prior,lines, final_direc, dir_data = data_direc, xtype = xtype, naming_convention="_data_struct_2020_10_02.npy")
"""
