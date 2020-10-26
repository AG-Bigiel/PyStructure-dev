import numpy as np
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt



def stack_spec(spec_ra,
                x, \
                xtype, \
                xmin_in = None, \
                xmax_in = None, \
                binsize_in = None,\
                irregular = False,\
                mask =None,\
                use_median = False,\
                calc_scatter = False):

    # Number of channels
    n_vaxis = len(spec_ra[0])
    # Number of lines of sight
    n_vec = len(spec_ra)
    spec_array = np.zeros((n_vec, n_vaxis))
    for i in range(n_vec):
        spec_array[i, :] = spec_ra[i]
    spec_ra = spec_array
    if mask is None:
        mask = np.array(~np.isnan(x), dtype = int)

    #--------------------------------------------------------------------------
    # Construct the bins
    #--------------------------------------------------------------------------

    if xtype in ['sfr',"12co10","12co21","PACS", "sigtir", "TIR_co21", "TIR_co10"]:
        xmax = xmax_in
        xmin = xmin_in
        deltax = (xmax - xmin)
        if binsize_in is None:
            #    ... DEFAULT TO 10 BINS
            print("[WARNING]\t No binsize specified. Making 10 bins.")
            binsize = deltax / 10.
        else:
            binsize = binsize_in
    #    MAKE THE BINS
        nbins = int(abs(np.ceil(deltax / binsize)))
        xmin_bin = 10**(np.arange(nbins)*binsize)*10**xmin
        xmax_bin = xmin_bin*10**binsize #< 10**xmax_in
        xmid_bin = xmin_bin*10**(binsize*0.5)
    else:
        #WORK OUT THE MINIMUM FOR THE BINS
        if xmin_in == None:
            xmin = np.nanmin(x)
        else:
            xmin = xmin_in
        if xmax_in == None:
            xmax = np.nanmax(x)
        else:
            xmax = xmax_in
        if not irregular:
            deltax = (xmax - xmin)
            if binsize_in is None:
                #    ... DEFAULT TO 10 BINS
                print("[WARNING]\t No binsize specified. Making 10 bins.")
                binsize = deltax / 10.
            else:
                binsize = binsize_in
            #;    MAKE THE BINS
            nbins = int(abs(np.ceil(deltax / binsize)))
            xmin_bin = np.arange(nbins)*binsize+xmin
            xmax_bin = xmin_bin+binsize #< xmax
            xmid_bin = (xmin_bin+xmax_bin)*0.5


        else:
            nbins = len(xmax_in)
            xmin_bin = xmin_in
            xmax_bin = xmax_bin
            xmid_bin = (xmin_bin+xmax_bin)*0.5

    #--------------------------------------------------------------------------
    # Initializze the output
    #--------------------------------------------------------------------------
    stacked_spec = np.zeros((n_vaxis, nbins))*np.nan
    if calc_scatter:
        scatter_spec = np.zeros((n_vaxis, nbins))*np.nan
    counts = np.zeros((n_vaxis, nbins))


    #--------------------------------------------------------------------------
    # Loop over bins
    #--------------------------------------------------------------------------
    
    for i in range(nbins):	
        
        binind = np.where(np.logical_and(np.logical_and(x >= xmin_bin[i],x<xmax_bin[i]),\
                          mask == 1))[0]
        binct = len(binind)
       
        if binct == 0:
            continue
        elif binct == 1:
            stacked_spec[:,i] = spec_ra[binind,:]
            counts[:,i] = np.isfinite(spec_ra[binind,:])
            continue
        else:
            
            # Average togethre all the SPECTRA
            image_here = spec_ra[binind,:]
            counts[:,i] = np.sum(np.isfinite(spec_ra[binind,:]), axis = 0)
            
            if use_median:
                stacked_spec[:,i] = np.nanmedian(image_here, axis = 0)
                if calc_scatter:
                    for k in range(n_vaxis):
                        scatter_spec[k,i] = median_absolute_deviation(image_here[:,k])
            else:
                stacked_spec[:,i] = np.nansum(image_here, axis = 0)/counts[:,i]
                if calc_scatter:
                    for k in range(n_vaxis):
                        scatter_spec[k,i] = np.nanstd(image_here[:,k])
            
    #--------------------------------------------------------------------------
    # Return Stacked Spectrum
    #--------------------------------------------------------------------------
    output = {}
    output["spec"] = stacked_spec
        
    if calc_scatter:
        output["scatt"] = scatter_spec
    output["counts"] = counts
    output["xmid"] = xmid_bin
    
    return output
