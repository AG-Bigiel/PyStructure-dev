import numpy as np
import pandas as pd
import sys
#import binning as bin_func
import bin_profile as prof
import new_stack as new_stacks


def stack_pix_by_x(struct, lines, xtype, x_vec , xmin_bin, xmax_bin, xmid_bin,mask = None, median = "mean", out_struc_in = None):


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

    # -------------------------------------------------------------------------
    # CONSTRUCT THE BINS
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Loop Over List of Tag Names
    # ------------------------------------------------------------------------

    out_struc['xtype'] = xtype
    out_struc['xmin'] = xmin_bin
    out_struc['xmax'] = xmax_bin
    out_struc['xmid'] = xmid_bin
    return out_struc
