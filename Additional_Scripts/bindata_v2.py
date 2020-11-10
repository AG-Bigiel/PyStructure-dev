import numpy as np

#### Uncertainty
def sigma_err(sigma_data, oversample):

    '''Calculate uncertainy for bin considering only y data'''
    id_nonnan = ~np.isnan(sigma_data)
    sigma_data_ = sigma_data[id_nonnan]
    n = len(sigma_data_)

    sigma_err_var = np.sqrt(np.nansum(sigma_data_**2 / n))
    sigma_err_mean = sigma_err_var / np.sqrt(n / oversample)
    sigma_err_median = sigma_err_mean * 1.25

    return sigma_err_mean, sigma_err_median

#not used yet...
def sigma_err_2d(data1_err, data2_err):

    '''Calculate uncertainy for bin considering both x and y data'''

    n = len(data1_err)

    sigma1_err_ = np.nansum(data1_err**2)**0.5 / n
    sigma2_err_ = np.nansum(data2_err**2)**0.5 / n
    sigma_err = (sigma1_err_**2 + sigma2_err_**2)**0.5

    return sigma_err
####

### Get binned y data, with defined bins in x data
def get_bins_1d(data1, data2, data1_err, data2_err, oversample=1, bins='', nbins=10, logbins=False, method='median'):

    '''Takes two numpy arrays of data and computes the binned data.
        INPUT:
            data1 = x-axis data - accepts multi-dimentional arrays
            data2 = y-axis data - accepts multi-dimentional arrays
            data1_err = x-axis data err (not used)
            data2_err = y-axis data err
            oversample = data oversampling ratio, accounting that data points oversample beam, and are therefore correlated and not independent measuremnets- usually N_oversample = 1.13 * beam_area / pixel_area; Default = 1
            bins = bin ranges, i.e. edges of bins; Default = determine bins from min and max x-data values
            nbins = if bins not defined, then number of bins to deteremine; Default = 10
            logbins = if bins not defined, log spaced bins for data >0
            method = mean or median, default = median
        OUTPUT:
            x = mean or median binned data x
            y = mean or median binned data y
            stats = statistics of binned data:                
                ['significance'] = signficance of each bin (i.e. S/N) - only y-data error taken into account
                ['ybin_err'] = error in each bin - only y-data error taken into account
                ['-1sigma'] = -1 sigma (15.9 percentile) of y-data within bin
                ['+1sigma'] = +1 sigma (84.1 percentile) of y-data within bins
                ['nneg'] = number of negative data points within with bins (useful for plotting)
                ['npos'] = number of posative data points within with bins (useful for plotting)
                ['ntot'] = number of all data points within with bins (useful for plotting)
                ['isnan'] = number of nan data points within with bins (useful for error checks)
                ['isnotnan'] = number of non-nan data points within with bins (useful for error checks)
            bins = bin ranges, i.e. edges of bins
        '''

    data1 = data1.flatten()
    data2 = data2.flatten()
    data1_err = data1_err.flatten()
    data2_err = data2_err.flatten()
    
    #Remove nan values in data2 -> may screw with stats!
    id_nonnan = np.where(~np.isnan(data2))
    data1 = data1[id_nonnan]
    data2 = data2[id_nonnan]
    data1_err = data1_err[id_nonnan]
    data2_err = data2_err[id_nonnan]
    
    id_nonnan = np.where(~np.isnan(data2_err))
    data1 = data1[id_nonnan]
    data2 = data2[id_nonnan]
    data1_err = data1_err[id_nonnan]
    data2_err = data2_err[id_nonnan]

    if bins=='':
        min=np.nanmin(data1)  # COMMENT (L.N.): this overwrites the default min function (I would not recommend doing that)
        max=np.nanmax(data1)  # COMMENT (L.N.): s. above
        if logbins:
            min = np.nanmin(data1[data1>0])
            bins = np.logspace(np.log10(min), np.log10(max), nbins+1)
        else:
            bins = np.linspace(min, max, nbins+1)
    else:
        nbins = len(bins)-1

    x = np.empty([nbins]) *np.nan
    y = np.empty([nbins]) *np.nan
    p1 = np.empty([nbins]) *np.nan
    p2 = np.empty([nbins]) *np.nan

    sigma = np.empty([nbins]) *np.nan
    significant = np.empty([nbins]) *np.nan
    neg = np.empty([nbins]) *np.nan
    pos = np.empty([nbins]) *np.nan
    ntot = np.empty([nbins]) *np.nan
    isnan = np.empty([nbins]) *np.nan
    isnotnan = np.empty([nbins]) *np.nan
    
    for i in range(nbins):

        ids_ = (data1 > bins[i]) & (data1 < bins[i+1])
        ids = np.where(ids_)

        if method == 'median':
            y[i] = np.nanmedian(data2[ids])
            x[i] = np.nanmedian([data1[ids], data1[ids]])  # COMMENT (L.N.): Why a list of arrays?
        elif method == 'mean':
            y[i] = np.nanmean(data2[ids])
            x[i] = np.nanmean([data1[ids], data1[ids]])
        else:
            print('[ERROR] Method not known')
            
        x[i] = (bins[i+1]+bins[i])/2
        p1[i] = np.nanpercentile(data2[ids], 50 - 34.1)
        p2[i] = np.nanpercentile(data2[ids], 50 + 34.1)

        sigma_err_mean, sigma_err_median = sigma_err(data2_err[ids], oversample)

        if method == 'median':
            sigma[i] = sigma_err_median
        elif method == 'mean':
            sigma[i] = sigma_err_mean
            
        significant[i] = y[i] / sigma[i]

        neg[i] = len(np.where(data2[ids]<=0)[0])
        pos[i] = len(np.where(data2[ids]>0)[0])
        ntot[i] = len(data2[ids])
        isnan[i] = len(np.where(np.isnan(data2[ids]))[0])
        isnotnan[i] = len(np.where(~np.isnan(data2[ids]))[0])
        
        if isnan[i] > 0: 
            print('[Warning] Bin contains nan values that should have been taken care of!')
            
    id_ = ~np.isnan(significant)
    significant = significant[id_]
    sigma = sigma[id_]
    p1 = p1[id_]
    p2 = p2[id_]
    x = x[id_]
    y = y[id_]
    ntot = ntot[id_]
    isnan = isnan[id_]
    isnotnan = isnotnan[id_]
    neg = neg[id_]
    pos = pos[id_]
    
    keys = ['significance', 'ybin_err', '-1sigma', '+1sigma', 'nneg', 'npos', 'ntot', 'isnotnan', 'isnan']

    stats = dict.fromkeys(keys)

    stats['significance'] = significant
    stats['ybin_err'] = sigma
    stats['-1sigma'] = p1
    stats['+1sigma'] = p2
    stats['nneg'] = neg
    stats['npos'] = pos
    stats['ntot'] = ntot
    stats['isnan'] = isnan
    stats['isnotnan'] = isnotnan
    

    return x, y, stats, bins


### Get binned ratio y1/y2 data, with defined bins in x data
def get_bins_ratio_1d(xdata, y1data, y2data, xdata_err, y1data_err, y2data_err, oversample=1, bins='', nbins=10, logbins=False, method='median'):

    '''Takes three numpy arrays of data and computes the binned data of the ratio of the two latter.
       This is made for binning of the kind y=y1/y2 against x, where y1 and y2 are binned indivdually in x.
        INPUT:
            xdata = x-axis data - accepts multi-dimentional arrays
            y1data = nomenator y-axis data - accepts multi-dimentional arrays
            y2data = denomenator y-axis data - accepts multi-dimentional arrays, must have same shape as y1
            xdata_err = x-axis data err (not used)
            y1data_err = nomenator y-axis data err
            y2data_err = denomenator y-axis data err
            oversample = data oversampling ratio, accounting that data points oversample beam, and are therefore correlated and not independent measuremnets- usually N_oversample = 1.13 * beam_area / pixel_area; Default = 1
            bins = bin ranges, i.e. edges of bins; Default = determine bins from min and max x-data values
            nbins = if bins not defined, then number of bins to deteremine; Default = 10
            logbins = if bins not defined, log spaced bins for data >0
            method = mean or median, default = median
        OUTPUT:
            x = mean or median binned data x
            y = mean or median binned data y = bin(y1)/bin(y2)
            stats = statistics of binned data:                
                ['significance'] = signficance of each bin (i.e. S/N) - only y-data error taken into account
                ['ybin_err'] = error in each bin - only y-data error taken into account
                ['-1sigma'] = -1 sigma (15.9 percentile) of y-data within bin  # TBD
                ['+1sigma'] = +1 sigma (84.1 percentile) of y-data within bins  # TBD
                ['nneg'] = number of negative data points within with bins (useful for plotting)
                ['npos'] = number of posative data points within with bins (useful for plotting)
                ['ntot'] = number of all data points within with bins (useful for plotting)
                ['isnan'] = number of nan data points within with bins (useful for error checks)
                ['isnotnan'] = number of non-nan data points within with bins (useful for error checks)
            bins = bin ranges, i.e. edges of bins
        '''

    xdata = xdata.flatten()
    y1data = y1data.flatten()
    y2data = y2data.flatten()
    xdata_err = xdata_err.flatten()
    y1data_err = y1data_err.flatten()
    y2data_err = y2data_err.flatten()
    
    # check if y1 is compatible with y2
    if len(y1data) != len(y2data):
        print('[ERROR] y1 data is not compatible with y2 data!')
    
    #Remove nan values in y data -> may screw with stats!
    id_nonnan = np.where(~np.isnan(y1data) & ~np.isnan(y2data) & ~np.isnan(y1data_err) & ~np.isnan(y2data_err))
    xdata = xdata[id_nonnan]
    y1data = y1data[id_nonnan]
    y2data = y2data[id_nonnan]
    xdata_err = xdata_err[id_nonnan]
    y1data_err = y1data_err[id_nonnan]
    y2data_err = y2data_err[id_nonnan]

    if bins=='':
        xmin=np.nanmin(xdata)
        xmax=np.nanmax(xdata)
        if logbins:
            xmin = np.nanmin(xdata[xdata>0])
            bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
        else:
            bins = np.linspace(xmin, xmax, nbins+1)
    else:
        nbins = len(bins)-1

    # make empty arrays for bin results and statistics
    x = np.empty([nbins]) *np.nan
    y1 = np.empty([nbins]) *np.nan
    y2 = np.empty([nbins]) *np.nan
    y = np.empty([nbins]) *np.nan  # ratio y = y1/y2
    p1 = np.empty([nbins]) *np.nan
    p2 = np.empty([nbins]) *np.nan

    sigma = np.empty([nbins]) *np.nan
    significant = np.empty([nbins]) *np.nan
    neg = np.empty([nbins]) *np.nan
    pos = np.empty([nbins]) *np.nan
    ntot = np.empty([nbins]) *np.nan
    isnan = np.empty([nbins]) *np.nan
    isnotnan = np.empty([nbins]) *np.nan
    
    # loop over bins
    for i in range(nbins):

        # get indices of data inside the current bin
        ids_ = (xdata > bins[i]) & (xdata < bins[i+1])
        ids = np.where(ids_)

        # get bin medians
        if method == 'median':
            x[i] = np.nanmedian(xdata[ids])
            y1[i] = np.nanmedian(y1data[ids])
            y2[i] = np.nanmedian(y2data[ids])
            y[i] = y1[i]/y2[i]
            
        # get bin means
        elif method == 'mean':
            x[i] = np.nanmean(xdata[ids])
            y1[i] = np.nanmean(y1data[ids])
            y2[i] = np.nanmean(y2data[ids])
            y[i] = y1[i]/y2[i]
            
        else:
            print('[ERROR] Method not known')
            
        x[i] = (bins[i+1]+bins[i])/2
        #p1[i] = np.nanpercentile(data2[ids], 50 - 34.1)  # not implemented yet
        #p2[i] = np.nanpercentile(data2[ids], 50 + 34.1)  # s. above

        sigma1_err_mean, sigma1_err_median = sigma_err(y1data_err[ids], oversample)
        sigma2_err_mean, sigma2_err_median = sigma_err(y2data_err[ids], oversample)

        # get bin median uncertainties
        if method == 'median':
            sigma[i] = np.sqrt((1/y2[i] * sigma1_err_median)**2 + (y1[i]/(y2[i]**2) * sigma2_err_median)**2)  # Gaussian error propagation
            
        # get bin mean uncertainties
        elif method == 'mean':
            sigma[i] = np.sqrt((1/y2[i] * sigma1_err_mean)**2 + (y1[i]/(y2[i]**2) * sigma2_err_mean)**2)  # Gaussian error propagation
            
        significant[i] = y[i] / sigma[i]  # signal-to-noise ratio

        neg[i] = len(np.where( (y1data[ids]<=0) | (y2data[ids]<=0) )[0])
        pos[i] = len(np.where( (y1data[ids]>0) | (y2data[ids]>0) )[0])
        ntot[i] = len(y1data[ids])
        isnan[i] = len(np.where(np.isnan(y1data[ids]) | np.isnan(y2data[ids]))[0])
        isnotnan[i] = len(np.where(~np.isnan(y1data[ids]) & ~np.isnan(y2data[ids]))[0])
        
        if isnan[i] > 0: 
            print('[Warning] Bin contains nan values that should have been taken care of!')
            
    id_ = ~np.isnan(significant)  # indices where signal-to-noise ratio is a valid number
    significant = significant[id_]
    sigma = sigma[id_]
    #p1 = p1[id_]
    #p2 = p2[id_]
    x = x[id_]
    y = y[id_]
    ntot = ntot[id_]
    isnan = isnan[id_]
    isnotnan = isnotnan[id_]
    neg = neg[id_]
    pos = pos[id_]
    
    keys = ['significance', 'ybin_err', '-1sigma', '+1sigma', 'nneg', 'npos', 'ntot', 'isnotnan', 'isnan']

    stats = dict.fromkeys(keys)

    stats['significance'] = significant
    stats['ybin_err'] = sigma
    #stats['-1sigma'] = p1
    #stats['+1sigma'] = p2
    stats['nneg'] = neg
    stats['npos'] = pos
    stats['ntot'] = ntot
    stats['isnan'] = isnan
    stats['isnotnan'] = isnotnan
    

    return x, y, stats, bins