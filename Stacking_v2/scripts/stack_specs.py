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
                ignore_empties=False,
                trim_stackspec = False,
                spec_orig = None):

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
    
    #count the number of pixels per bin
    counts_tot = np.zeros(nbins)
    
    #count the number of pixels per bin that contain an actual observation
    counts_tot_spec = np.zeros(nbins)

    #--------------------------------------------------------------------------
    # Loop over bins
    #--------------------------------------------------------------------------
    
    for i in range(nbins):	
        
        binind = np.where(np.logical_and(np.logical_and(x >= xmin_bin[i],x<xmax_bin[i]),\
                          mask == 1))[0]
        
        # get total number of spectra in each bin
        counts_tot[i] = len(binind)
        
        if counts_tot[i] == 0:
            continue
        elif counts_tot[i] == 1:
            stacked_spec[:,i] = spec_ra[binind,:]
            counts[:,i] = np.isfinite(spec_ra[binind,:])
            counts_tot_spec[i] = np.nansum(np.nansum(spec_orig[binind,:], axis=1)!=0)
            continue
        else:
            
            # Average together all the SPECTRA
            spec_list = spec_ra[binind,:]
            counts[:,i] = np.sum(np.isfinite(spec_ra[binind,:]), axis = 0)
            counts_tot_spec[i] = np.nansum(np.nansum(spec_orig[binind,:], axis=1)!=0)
            
            if use_median:
                stacked_spec[:,i] = np.nanmedian(spec_list, axis = 0)
              
            else:
                if weights is not None:
                    stacked_spec[:,i] = np.nansum(weights[binind,np.newaxis]*spec_list, axis = 0)/np.nansum(weights[binind])
                else:
                    if ignore_empties:
                        # divide by number of channels with detection of prior
                        stacked_spec[:,i] = np.nansum(spec_list, axis = 0)/counts[:,i]
                    else:
                        # divide by total number of counts
                        stacked_spec[:,i] = np.nansum(spec_list, axis = 0)/counts_tot_spec[i]
                
                if trim_stackspec:
                    # trim the edges of the spectrum to only include channels where the overlap
                    # of all spectrea is ensured
                    id_trim = np.where(counts[:,i]<np.nanmax(counts[:,i]))
                    
                    if len(id_trim[0])==len(stacked_spec[:,i]):
                        print("[WARNING]\tNo spectral overlap for certain spectra. Turn-off Trimming.")
                    stacked_spec[:,i][id_trim]=np.nan
                        
    #--------------------------------------------------------------------------
    # Return Stacked Spectrum
    #--------------------------------------------------------------------------
    output = {}
    output["spec"] = stacked_spec
    output["counts"] = counts
    output["counts_total"] = counts_tot
    output["counts_total_spec"] = counts_tot_spec
    output["xmid"] = xmid_bin
    
    return output
