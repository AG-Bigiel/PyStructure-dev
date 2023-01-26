import numpy as np



def get_bins(x,
            scaling,
            nbins,
            xmin_in=None,
            xmax_in=None,
            ):


    #--------------------------------------------------------------------------
    # Construct the bins
    #--------------------------------------------------------------------------
    #define minima and  maxima
    if xmin_in is None:
        if scaling in "linear":
            xmin = np.nanmin(x)
            print("[INFO]\t No minimum specified. Defaulting to data minimum: ", xmin)
        else:
            #if logscaling, ignore the negative values
            xmin = np.nanpercentile(x[x>0],10)
            print("[INFO]\t No minimum specified and log-scale. Defaulting to: ", xmin)
    else:
        xmin = xmin_in

    
    if xmax_in is None:
        xmax = np.nanmax(x)
        print("[INFO]\t No maximum specified. Defaulting to data maximum: ", xmax)
    else:
        xmax = xmax_in
            
    #--------------------------------------------------------------------------------
    
    if nbins is None:
        print("[INFO]\t No nbins specified. Making 10 bins.")
        nbins = 10
 

    
    # MAKE BINS
    
    if scaling in ["linear"]:
        deltax = xmax - xmin
        binsize = deltax/nbins
        xmin_bin = np.arange(nbins)*binsize + xmin
        xmax_bin = np.minimum(xmin_bin + binsize, xmax)
        xmid_bin = (xmin_bin + xmax_bin)*0.5
    
    elif scaling in ["log"]:
        deltax = np.log10(xmax) - np.log10(xmin)
        binsize = deltax/nbins
        
        xmin_bin = 10**(np.arange(nbins)*binsize)*xmin
        xmax_bin = xmin_bin*10**binsize #< 10**xmax_in
        xmid_bin = xmin_bin*10**(binsize*0.5)
    
    else:
        print("[Error]\t Bin-scaling not defined")
        return 0


    return nbins, xmin_bin, xmax_bin, xmid_bin
