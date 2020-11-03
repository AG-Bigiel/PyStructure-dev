import numpy as np
import pandas as pd
import sys
#import binning as bin_func
import bin_profile as prof
import new_stack as new_stacks

def stack_pix_by_x(struct, lines, xtype, x_vec ,xmin, xmax ,galaxy, binsize = None ,mask = None, median = "mean", out_struc_in = None):


    # -------------------------------------------------------------------------
    #DEFINE THE FLAG DEPENDING ON THE STACKING METHOD
    # -------------------------------------------------------------------------
    flag = xtype

    #HERE I HAVE TO ADD THE OTHER CASES DIFFERENT FROM RADIAL BINNING!

    #INITIALIZE OUTPUT
    if out_struc_in is None:
        out_struc = new_stacks.new_stacking(lines)
    else:
        out_struc = out_struc_in
    #MEASURE THE SIZES OF THIS THING
    n_vec = len(x_vec)

    #INITIALIZE THE MASK
    if mask is None:
        mask = np.array(~np.isnan(x_vec), dtype = int)

    # -------------------------------------------------------------------------
    # CONSTRUCT THE BINS
    # -------------------------------------------------------------------------

    # WORK OUT THE MAXIMUM FOR THE BINS

    #First construct the logarithmic bins for xtype = sfr

    if xtype in ['sfr',"12co10","12co21","PACS","sigtir","TIR_co21","TIR_co10"]:

        deltax = (xmax - xmin)
        if binsize is None:
            #    ... DEFAULT TO 10 BINS
            print("[WARNING]\t No binsize specified. Making 10 bins.")
            binsize = deltax / 10.

    #    MAKE THE BINS
        nbins = abs(np.ceil(deltax / binsize))
        xmin_bin = 10**(np.arange(nbins)*binsize)*10**xmin
        xmax_bin = xmin_bin*10**binsize #< 10**xmax_in
        xmid_bin = xmin_bin*10**(binsize*0.5)
    else:
        #WORK OUT THE MINIMUM FOR THE BINS

        deltax = (xmax - xmin)
        if binsize is None:
            #    ... DEFAULT TO 10 BINS
            print("[WARNING]\t No binsize specified. Making 10 bins.")
            binsize = deltax / 10.

        #;    MAKE THE BINS
        nbins = abs(np.ceil(deltax / binsize))
        xmin_bin = np.arange(nbins)*binsize+xmin
        xmax_bin = xmin_bin+binsize #< xmax
        xmid_bin = (xmin_bin+xmax_bin)*0.5

    # -------------------------------------------------------------------------
    # Loop Over List of Tag Names
    # ------------------------------------------------------------------------
    stack_tags = pd.read_csv("new_stack_rad_tags.txt", sep = "\t\t",comment = ";",engine='python')

    ntag = len(stack_tags["tag_in"])
    try:
        data_tags = struct.dtype.names
    except:
        data_tags = struct.keys()
    out_tags = list(out_struc.keys())

    maskind = np.where(mask)
    # Loop over tag names
    for i in range(ntag):

        if not stack_tags["tag_in"][i] in list(data_tags):
            continue
        else:
            bins_output = prof.bin_prof(x_vec[maskind], struct[stack_tags["tag_in"][i]][maskind],\
                                 xmin_in = xmin, xmax_in = xmax, binsize_in = binsize, oversamp = 1. )


        if median == "median":
            out_struc[stack_tags["tag_in"][i]] = bins_output["medprof"]
        else:
            out_struc[stack_tags["tag_in"][i]] = bins_output["meanprof"]
    out_struc['xtype'] = xtype
    out_struc['xmin'] = xmin_bin
    out_struc['xmax'] = xmax_bin
    out_struc['xmid'] = xmid_bin

    return out_struc
