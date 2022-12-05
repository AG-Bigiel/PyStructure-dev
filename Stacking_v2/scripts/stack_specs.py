import numpy as np
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt



def stack_spec(spec_ra,
                x, \
                xtype, \
                nbins, xmin_bin, xmax_bin, xmid_bin,\
                mask =None,\
                use_median = False,\
                weights = None,
                ignore_empties=False):

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
    # Initializze the output
    #--------------------------------------------------------------------------
    stacked_spec = np.zeros((n_vaxis, nbins))*np.nan
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
            
            # Average together all the SPECTRA
            image_here = spec_ra[binind,:]
            counts[:,i] = np.sum(np.isfinite(spec_ra[binind,:]), axis = 0)
            
            # get total number of spectra in each bin
            counts_tot = len(spec_ra[binind,:])
            
            if use_median:
                stacked_spec[:,i] = np.nanmedian(image_here, axis = 0)
              
            else:
                if weights is not None:
                    stacked_spec[:,i] = np.nansum(weights[binind,np.newaxis]*image_here, axis = 0)/np.nansum(weights[binind])
                else:
                    if ignore_empties:
                        # divide by number of channels with detection of prior
                        stacked_spec[:,i] = np.nansum(image_here, axis = 0)/counts[:,i]
                    else:
                        # divide by total number of counts
                        stacked_spec[:,i] = np.nansum(image_here, axis = 0)/counts_tot
                
            
    #--------------------------------------------------------------------------
    # Return Stacked Spectrum
    #--------------------------------------------------------------------------
    output = {}
    output["spec"] = stacked_spec

    output["counts"] = counts
    output["xmid"] = xmid_bin
    
    return output
