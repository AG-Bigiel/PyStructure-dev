import numpy as np
from astropy.stats import median_absolute_deviation


def bin_prof(x,y,  xmin_in = None,\
                xmax_in = None, \
               binsize_in = None, \
               irregular = False,\
               oversamp = 1., \
               percentile = False):
    """
    bin_prof

    PURPOSE:

    Eventually, a generic routine for extracting binned profiles. Pretty damn
    close right now.

    CATEGORY:

    Science tool. Documentation to follow.

    CALLING SEQUENCE:

    bin_prof, x, y $
                , xmin = xmin_in, xmax = xmax_in $
                , binsize=binsize_in $
                , medprof=medprof, meanprof=meanprof $
                , madprof=madprof, madlogprof=madlogprof $
                , stdprof=stdprof, errprof=errprof $
                , maxprof=maxprof, minprof=minprof $
                , countprof=countprof, oversamp=overamp $
                , percentile=percentile, lopercprof=lopercprof $
                , hipercprof=hipercprof
    INPUTS:


    OPTIONAL INPUTS:



    KEYWORD PARAMETERS:



    OPTIONAL OUTPUTS:



    PROCEDURE:



    EXAMPLE:

    MODIFICATION HISTORY:

    Added MIN/MAX measurement

    Converted form IDL to python by J. den Brok,

    """

    #--------------------------------------------------------------------------
    # Error Checking
    #--------------------------------------------------------------------------
    if len(x) != len(y):
        print("[ERROR]\t Missmatched of data, returning.")
        return


    #--------------------------------------------------------------------------
    # Construct the bins
    #--------------------------------------------------------------------------

    if xmax_in is None:
        if irregular:
            print("[ERROR]\t Irregular specified without bin maxima.")
            return
        print("[INFO]\t No maximum specified. Defaulting to Data,")
        xmax = np.nanmax(x)
    else:
        xmax = xmax_in

    if xmin_in is None:
        if irregular:
            print("[ERROR]\t Irregular specified without bin minima.")
            return
        print("[INFO]\t No maximum specified. Defaulting to Data,")
        xmin = np.nanmin(x)
    else:
        xmin = xmin_in


    if not irregular:
        deltax = xmax - xmin
        if binsize_in is None:
            print("[INFO]\t No binsize specified. Making 10 bins")
            binsize = deltax/10
        else:
            binsize = binsize_in
        #Make the bins
        # MAKE BINS
        nbins = int(np.ceil(deltax / binsize))
        xmin_bin = np.arange(nbins)*binsize + xmin

        xmax_bin = np.minimum(xmin_bin + binsize, xmax)
        xmid_bin = (xmin_bin + xmax_bin)*0.5

    else:
        nbins = len(x_max_in)
        xmin_bin = xmin
        xmax_bin = xmax
        xmid_bin = (xmin_bin + xmax_bin)*0.5
        xgeo_bin = np.sqrt(xmin_bin*xmax_bin)



    #--------------------------------------------------------------------------
    # Initialize Profiles
    #--------------------------------------------------------------------------

    #Initialize the output and fill with nans
    output = {}

    output["xmeanprof"] = np.zeros(nbins)*np.nan
    output["xmedianprof"] = np.zeros(nbins)*np.nan
    output["medprof"] = np.zeros(nbins)*np.nan
    output["meanprof"] = np.zeros(nbins)*np.nan
    output["madprof"] = np.zeros(nbins)*np.nan
    output["madlogprof"] = np.zeros(nbins)*np.nan
    output["stdprof"] = np.zeros(nbins)*np.nan
    output["errprof"] = np.zeros(nbins)*np.nan
    output["maxprof"] = np.zeros(nbins)*np.nan
    output["minprof"] = np.zeros(nbins)*np.nan
    output["countprof"] = np.zeros(nbins)*np.nan

    if percentile:
        output["hipercprof"] = np.zeros(nbins)*np.nan
        output["lopercprof"] = np.zeros(nbins)*np.nan


    #--------------------------------------------------------------------------
    # Generate the profiles
    #--------------------------------------------------------------------------
    for ii in range(nbins):

        # Find the pixels in the present ring
        binind = np.where(np.logical_and(x>xmin_bin[ii], x<xmax_bin[ii]))
        binct = len(binind[0])

        # note the counts in the counts profile
        output["countprof"][ii] = binct
        if binct>1:
            indep = x[binind]
            data = y[binind]

            # ... Mean and median Value
            output["xmeanprof"][ii] = np.nanmean(indep)
            output["xmedianprof"][ii] = np.nanmedian(indep)
            output["meanprof"][ii] = np.nanmean(data)
            output["medprof"][ii] = np.nanmedian(data)

            # ... Max and Minimum data
            output["minprof"][ii] = np.nanmin(data)
            output["maxprof"][ii] = np.nanmax(data)

            # Uncertaintz Estimates
            if binct>5:
                output["madprof"][ii]  = median_absolute_deviation(data)
                output["madlogprof"][ii]  = median_absolute_deviation(np.log10(data))
                output["stdprof"] [ii]= np.nanstd(data)
                output["errprof"][ii] = output["stdprof"] [ii]/np.sqrt(binct/oversamp)
            if percentile:
                data_sorted = sorted(data)
                perc = 0.5
                loind = np.ceil(binct*(0.5-perc/2))-1
                hiind = np.ceil(binct*(-.5+perc/2))-1
                output["lopercprof"][ii] = data[loind]
                output["hipercprof"][ii] = data[hiind]


    #--------------------------------------------------------------------------
    # Retunr the output array
    #--------------------------------------------------------------------------
    return output
