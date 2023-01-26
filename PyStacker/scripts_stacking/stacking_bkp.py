import numpy as np
import idlsave
import pandas as pd
import stacking_func as func
import stack_specs as stck_spc
import fitting_routine as fit
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation

import warnings
warnings.filterwarnings("ignore")





def get_stack(galaxy,prior_lines,lines, dir_save, dir_data ='./../../data/Database/', show = False, do_smooth = False, xtype = None, naming_convention = "_database.idl", sn_limits = [4,6], no_detec_wdw = 30):
    """
    Function converted from IDL to python
    :param galaxy: String of name of the galaxy, e.g. "ngc5194"
    :param dir_save: String of directory where to save the output
    :param dir_data: String of path to the directory, where the IDL database structure is saved
    :param show: show plots [not yet fully implemented]
    :param do_smooth: [variable not yet fully implemented]
    :param xtype: string name of the quantity by which to stack, must be one of the following: (rad, sfr, co21, co10, PACS, sigtir, TIR_co10, TIR_co21). If none is given, the program asks for user input in the command prompt.
    :param naming_convention: Naming of the idl structure file. E.g. if files are called "ngc5194_database.py, the convention is "_database.py".
    :param no_detec_wdw: window size over which to integrate in case no detection is found. In km/s
    """

    # --------------------------------------------------------------------
    # CHOOSE STACKING METHOD (In principle only the "rad" method is valid
    # now because otherwise more information in the structures is needed)
    # --------------------------------------------------------------------
    # Direcory, where the

    if xtype is None:
        xtype = input("Choose the stacking method (rad, sfr, co21, co10, PACS, sigtir, TIR_co10, TIR_co21): ")

        iter = 0
        while xtype not in ["rad", "sfr", "co21","co10","PACS","sigtir", "TIR_co10", "TIR_co21"] and iter <2:
            print('"'+xtype+'" is an invalid input. Please try again.')

            xtype = input("Choose the stacking method (rad, sfr, co21, co10, PACS, sigtir, TIR_co10, TIR_co21): ")
            iter +=1

        if iter == 2:
            print("Dude (m./f.), read the instructions!!!!")
            return None
    else:
        print("[INFO]\t Stacking by "+xtype)
    # -----------------------------------------------------------------------
    # DEFINE THE BINS DEPENDING ON WHICH STACKING METHOD IS USED
    # -----------------------------------------------------------------------
    bin_values = pd.read_csv("bin_values.txt", sep = "\t\t",comment = "#", engine = "python")

    flag_index = np.where(np.array(bin_values["FLAG"]) == xtype)[0]
    xmin=bin_values["VAL"][flag_index[0]]
    xmax=bin_values["VAL"][flag_index[1]]
    binsize = bin_values["VAL"][flag_index[2]]

    # -----------------------------------------------------------------------
    # RESTORE THE DATA STRUCTURES
    # -----------------------------------------------------------------------
    # Measurements
    #dir='/vol/alcina/data1/jdenbrok/Proj_I_2019/ngc_5194_database/measurements/'
    
    dir = dir_data
    file = dir + galaxy + naming_convention
    
    is_IDL = False
    if ".idl" in naming_convention:
        struct = idlsave.read(file, verbose=False, python_dict=True)
        this_data = struct["this_data"]
        is_IDL = True
    elif ".npy" in naming_convention:
        this_data = np.load(file,allow_pickle = True).item()
    else:
        raise TypeError("Input Database must me of format .idl or .npy")

    # -----------------------------------------------------------------------
    # LOOP THROUGH THE DIFFERENT GALAXIES
    # -----------------------------------------------------------------------

    if hasattr(galaxy, '__len__') == False or isinstance(galaxy,str):
        name=[galaxy]
    else:
        name = galaxy

    for i in range(len(name)):

    # determine xtype for this galaxy -> x axis !!!!!!!!!!!!!!!!!
        if xtype == 'rad':
            xvec = this_data["rgal_kpc"]
        elif xtype == 'sfr':
            path_sfr = dir_data+galaxy+"_sig_SFR_31as.txt"
            xvec = np.loadtxt(path_sfr)
        elif xtype == '12co21':
            xvec = this_data["INT_VAL_12CO21"]
        elif xtype == '12co10':
            xvec = this_data["INT_VAL_12CO10"]
        elif xtype == 'PACS':
            xvec = this_data["INT_VAL_PACS70"]/this_data["INT_VAL_PACS160"]
        elif xtype == "sigtir":
            path_sigtir = dir_data +galaxy+"_sig_TIR_31as.txt"
            xvec = np.loadtxt(path_sigtir)
        elif xtype == "TIR_co10":
            path_sigtir = dir_data +galaxy+"_sig_TIR_31as.txt"
            xvec = np.loadtxt(path_sigtir)/this_data["INT_VAL_12CO10"]
        elif xtype == "TIR_co21":
            path_sigtir = dir_data +galaxy+"_sig_TIR_31as.txt"
            xvec = np.loadtxt(path_sigtir)/this_data["INT_VAL_12CO21"]
        elif xtype == "angle":
            xvec = this_data["theta_rad"]
        
        ra = this_data["ra_deg"]
        decl = this_data["dec_deg"]
        
            
        # -----------------------------------------------------------------------
        # AVERAGE PIXEL MEASUREMENTS
        # -----------------------------------------------------------------------

        #collect all pixels with xval value
        stack = func.stack_pix_by_x(this_data,lines, xtype, xvec ,xmin, xmax , galaxy, binsize , median  = 'mean' ) #0= mean, 1=median
        
        #--------------------------------------------------------------------------
        # STACK THE SPECTRA
        # -------------------------------------------------------------------------

        # stack spectra in those pixels
        shuffled_specs = {}
        
        #length of the shuffeled spectrum
        nvaxis = len(this_data["SPEC_VAL_SHUFF"+prior_lines[0]][0])
        for line in lines+prior_lines:
            shuffled_specs["SPEC_VAL_SHUFF"+line] = this_data["SPEC_VAL_SHUFF"+line]

        
        # create a mask and combine for co21 and co10 at least:
        spec_co21 = shuffled_specs["SPEC_VAL_SHUFF"+prior_lines[0]]
        

        idnans_co21 = np.ones(len(spec_co21))
       
        for i in range(len(idnans_co21)):
            if not sum(np.array(np.isnan(spec_co21[i]), dtype = int)) == nvaxis:
                idnans_co21[i] = 0
            
        comb_mask = np.array(idnans_co21, dtype=int)

        for i in range(len(comb_mask)):
            if comb_mask[i] == 1:
                spec_co21[i] = np.zeros(nvaxis)*np.nan
               
        shuffled_specs[prior_lines[0]+"_spec_K"]  = spec_co21
        
        
        if do_smooth == False:
            if is_IDL:
                vaxis = this_data[0]["SPEC_VCHAN0_SHUFF"+prior_lines[0]] + this_data[0]["SPEC_DELTAV_SHUFF"+prior_lines[0]] * np.arange(len(this_data[0]["SPEC_VAL_"+prior_lines[0]]))
            else:
                vaxis = this_data["SPEC_VCHAN0_SHUFF"+prior_lines[0]] + this_data["SPEC-DELTAV_SHUFF"+prior_lines[0]] * np.arange(len(this_data["SPEC_VAL_SHUFF"+prior_lines[0]][0]))
            co21_spec = shuffled_specs["SPEC_VAL_SHUFF"+prior_lines[0]]

            stack_spec = stck_spc.stack_spec(co21_spec, xvec,xtype, xmin, xmax, binsize)
            stack[prior_lines[0]+"_spec_K"] = stack_spec["spec"]

            # Iterate over the different lines that need to be stacked

            for line in lines:
                spec = shuffled_specs["SPEC_VAL_SHUFF"+line]
                stack_spec = stck_spc.stack_spec(spec, xvec,xtype, xmin, xmax, binsize)
                stack[line+"_spec_K"] = stack_spec["spec"]

              
                #print(np.shape(stack_spec["spec"]))

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
    stack["gal"] = galaxy
    stack["vaxis_kms"] = vaxis

    stack["ncounts"] = np.nanmax(stack_spec["counts"], axis = 0)
    stack["narea_kpc2"] = 37.575*(1/3600*stack["dist_mpc"]*np.pi/180)**2* \
                          np.cos(np.pi/180*stack["incl_deg"])*stack["ncounts"]


    
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
        stack["limit_K_kms_"+line] = np.zeros(n_stacks)*np.nan
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
            
            prior = stack[prior_line+"_spec_K"][:,j]
            
            #Estimate rms
            rms = median_absolute_deviation(prior, axis = None,ignore_nan=True)
            rms = median_absolute_deviation(prior[np.where(prior<3*rms)],ignore_nan=True)
            
            
            # Mask each spectrum
            mask = np.array(prior > sn_up * rms, dtype = int)
            low_mask = np.array(prior > sn_low * rms, dtype = int)
       
            mask = mask & (np.roll(mask, 1) | np.roll(mask,-1))
        
            #expand to cover all > 2sigma that have a 2-at-4sigma core
            for kk in range(5):
                mask = np.array(((mask + np.roll(mask, 1) + np.roll(mask, -1)) >= 1), dtype = int)*low_mask

            mask = np.array((mask + np.roll(mask, 1) + np.roll(mask, -1))>=1, dtype = int)
            
            prior_specs.append(prior)
            masks.append(mask)
            rms_values.append(rms)

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
        
        #line_vmean = np.nansum(v * prior*mask)/ np.nansum(prior*mask)
        
        #prior_ii = np.nansum(prior*mask)*abs(v[1]-v[0])
        
        #prior_uc = max([1, np.sqrt(np.nansum(mask))])*rms*abs(v[1]-v[0])
        
        #stack["rms_K_"+prior_line][j] = rms
        #stack["peak_K_"+prior_line][j] = np.nanmax(prior)
        #stack["ii_K_kms_"+prior_line][j] = prior_ii
        #stack["uc_ii_K_kms_"+prior_line][j] = prior_uc
        
        
        """
        ToDo: If none of the priors shows a detection, the program should give back only an upper limit
        """
        for line in lines+prior_lines:
            
            
            # Fit the line
            spec_to_integrate = stack[line+"_spec_K"][:,j]

            spec_to_integrate[spec_to_integrate == 0] = np.nan
            
            #calculate rms of line
            rms_line = median_absolute_deviation(spec_to_integrate, axis = None,ignore_nan=True)
            rms_line = median_absolute_deviation(spec_to_integrate[np.where(prior<3*rms_line)],ignore_nan=True)
            
            line_ii = np.nansum(spec_to_integrate*mask)*abs(v[1]-v[0])
            line_uc = max([1, np.sqrt(np.nansum(mask))])*rms_line*abs(v[1]-v[0])

            # Fill in dictionary
            stack["rms_K_"+line][j] = rms_line
            stack["peak_K_"+line][j] = np.nanmax(line_ii)
       
            stack["ii_K_kms_"+line][j] = line_ii
            stack["uc_ii_K_kms_"+line][j] = line_uc
            
            SNR_line = line_ii/line_uc
            stack["SNR_"+line][j] = SNR_line
            if SNR_line<3:
                #if window found: integrate over that
                if np.nansum(mask)>2:
                    stack["limit_K_kms_"+line][j] = 2*rms*max([1, np.sqrt(np.nansum(mask))])*abs(v[1]-v[0])
                
                #if no window found: use (default) 30 km/s window
                else:
                    stack["limit_K_kms_"+line][j] = 2*rms*no_detec_wdw


    path_save = dir_save + galaxy+"_stack_"+xtype+".npy"
    np.save(path_save, stack)
    print("[INFO]\t Successfull Finished")

    return stack
    
    
   
   

