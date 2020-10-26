import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def double_horn_func(x, p1, p2, sig, amp, ctr):
    """
    p1 - power in the gaussian term
    p2 - power in the double-horn term
    sig - width
    amp - amplitude
    ctr - center
    """
    use_x = x - ctr

    term1 = (1./np.sqrt(sig*np.pi**0.5))*np.exp(-use_x**2/sig**2/2.)
    term2 = (1./np.sqrt(sig*np.pi**0.5))*(-1./np.sqrt(2) + np.sqrt(2) * \
          (use_x / sig)**2)*np.exp(-use_x**2/sig**2/2)

    model = amp*(p1*term1 + p2*term2) / np.sqrt(p1**2 + p2**2)

    return model



def gaussian(x, amp, mu, sig):
    return amp * np.exp(-np.power(x-mu,2)/(2 * np.power(sig, 2.)))

def fit_line(vaxis, \
             spec, \
             galaxy = None,\
             line_name = "",\
             xmid = None,\
             lw_min= None,\
             lw_max = None,
             pad_v = 100,
             oversamp = 100,
             limline = 100):
    """
    ;+
    NAME:

    fit_co_line

    PURPOSE:

    Fit a gaussian or double-horn profile to a spectrum of intensity
    vs. velocity. Tailored to the "shuffled" case to that it can leverage the
    expectation that the line is at zero velocity. Tries to pay particular
    attention to upper limits.

    CATEGORY:

    science tool

    CALLING SEQUENCE:



    INPUTS:



    OPTIONAL INPUTS:



    KEYWORD PARAMETERS:



    OUTPUTS:



    OPTIONAL OUTPUTS:



    COMMON BLOCKS:



    SIDE EFFECTS:



    RESTRICTIONS:



    PROCEDURE:



    EXAMPLE:




    MODIFICATION HISTORY:
    Original IDL code from M. Jimenez-Donaire
    Converted to Python by J. den Brok, 12.2019
    -
    """
    # -------------------------------------------------------------------------
    # DEFAULTS, DEFINITIONS, AND ERROR-CATCHING
    # -------------------------------------------------------------------------
    
    ctr_guess = 0
    ctr_min = -20
    ctr_max = 50

    if lw_min is None or lw_max is None:
        lw_min = 30
        lw_max = 400

    lw_guess = 10
    offs_guess = 10

    result_fit = {}
    if np.sum(np.isfinite(spec)) == 0:
        result_fit["yfit"] = np.nan*vaxis
        result_fit["spec_detec"] = np.nan*vaxis
        result_fit["resid"] = np.nan*vaxis
        result_fit["type"] = 'error'
        result_fit["ii"] = np.nan
        result_fit["e_ii"] = np.nan
        result_fit["e_ii_prop"] = np.nan
        result_fit["fwhm"] = np.nan
        result_fit["quality"] = np.nan
        result_fit["rms"] = np.nan
        result_fit["zero"] = np.nan
        result_fit["peak"] = np.nan
        result_fit["limit_fit"] = np.nan*vaxis
        result_fit["limit_ii"] = np.nan
        result_fit["coefs"] = np.zeros(5)*np.nan
        result_fit["e_coefs"] = np.zeros(5)*np.nan
        return result_fit

    # Threshold to switch to fitting the broad line profile
    broad_line_thresh = 150

    # Define qualities
    int_snr_good = 10.
    peak_snr_good = 5.

    int_snr_limit = 5.
    peak_snr_limit = 3.

    # Velocity Spacing
    vdelt = abs(vaxis[1] - vaxis[0])

    # Protect the very outer part

    min_v = np.nanmin(vaxis[np.where(np.isfinite(spec))])
    max_v = np.nanmax(vaxis[np.where(np.isfinite(spec))])

    #Keep the edges of the spectrum empty
   
    always_empty = np.isfinite(spec) & (np.array(vaxis < np.min([(min_v + pad_v),max_v]), dtype = int) | \
                  np.array(vaxis > np.max([(max_v - pad_v),0]), dtype = int))

    always_spec = always_empty | np.array(np.isfinite(spec)==0)

    #Define a 50km/s window to hold the line to start with
    empty = np.array(np.array(abs(vaxis)>limline, dtype = int) + always_empty>=1, dtype = int)
    line = np.array(empty == 0, dtype = int)
    

    # Note the peak
    peak = np.nanmax(spec[np.where(line)])
   
    """
    plt.plot(vaxis, spec)
    plt.plot(vaxis[np.where(empty)], spec[np.where(empty)])
    plt.show()
    """
    #---------------------------------------------------------------------------
    # Fit our two models
    #--------------------------------------------------------------------------
    type_fit = "gaussian"
    n_iters = 10

    limit_fit = np.nan*vaxis
    limit_ii = np.nan
   
    if peak<0:
        type_fit = "limit"
    
    for iter in range(n_iters):
        empty += always_empty
        empty = np.array(empty>=1, dtype = int)
        line = np.array(empty==0, dtype = int)
        
        #Estimate the rms
        rms = np.nanstd(spec[np.where(empty)])
        
        if iter == 0:
            zero = np.nanmean(spec[np.where(empty)])
            #Shift the spectrum by the offset
            #spec -=zero

        #----------------------------------------------------------------------
        # Fit a gaussian to the profile
        #
        # Start by fitting a gaussian to the profile. We do this as long as the line
        # is narrow and relatively high signal to noise. If the SNR drops too low or
        # the line gets too broad, we change cases
        #----------------------------------------------------------------------
      
        if type_fit == "gaussian":
            p0 = [np.nanmax([peak,1e-4]), ctr_guess, lw_guess]
           
            fit_x = vaxis[np.where(line)]
            fit_y = spec[np.where(line)]
            
            no_nans = np.isfinite(fit_y)
            fit_sig = rms * (1+abs(vaxis[np.where(line)])/max(vaxis[np.where(line)]))
            
            bounds=((0, ctr_min, lw_min),(2*abs(np.nanmax(fit_y)),ctr_max,lw_max))
            
            
            popt, pcov = curve_fit(gaussian, fit_x[no_nans], fit_y[no_nans], p0 = p0, sigma = fit_sig[no_nans],\
                                    bounds = bounds)
            

            yfit = gaussian(vaxis, *popt)
            red_chisq = np.nansum((fit_y - yfit[np.where(line)])**2/fit_sig**2)/ \
                        (np.sum(np.isfinite(fit_y))/oversamp - 4)
            coefs = popt
            e_coefs =  np.sqrt(np.diag(pcov))
        #--------------------------------------------------------------------------
        #  FIT A DOUBLE HORN PROFILE
        #
        # If the line is too broad, fit a double-horn profile instead. These are
        # better at describing galaxy centers or cases where V_HI != V_CO. Still not
        # perfect, though.
        #--------------------------------------------------------------------------

        elif type_fit == "doublehorn":
            # Initial guess

            
            dh_guess = [1., 1., lw_guess, 2*peak, ctr_guess]
            fit_x = vaxis[np.where(line)]
            fit_y = spec[np.where(line)]
            fit_sig = rms+0.0*fit_y

            bounds = ((0,0,lw_min/4,0,ctr_min),(2*np.max(fit_y),2*np.max(fit_y),\
                      4*lw_max,10*np.nanmax(spec),ctr_max))
            popt_horn, pcov_horn = curve_fit(double_horn_func, fit_x, fit_y, sigma = fit_sig, \
                                   bounds = bounds)
            
            yfit = double_horn_func(vaxis, *popt_horn)
            yfit = np.maximum(yfit, 0)

            red_chisq = np.nansum((fit_y - yfit[np.where(line)])**2/fit_sig**2)/ \
                        (np.sum(np.isfinite(fit_y))/oversamp - 4)

            coefs = popt_horn
            e_coefs =  np.sqrt(np.diag(pcov_horn))

        #----------------------------------------------------------------------------
        # PLACE AN UPPER LIMIT
        #
        # In the low SNR case, heavily constrain the fit to get a best estimate of the
        # gaussian line that could be lurking in the data. Then work out what
        # amplitude causes this fit to become very unlikely for the spectrum.
        #
        #----------------------------------------------------------------------------

        elif type_fit == "limit":

            p0 = [np.max([np.nanmax(spec[np.where(line)])/2,0]), 0, 10]

            fit_x = vaxis[np.where(line)]
            fit_y = spec[np.where(line)]
            fit_sig = rms+0.0*fit_y


            bounds = ((0,-10,10),(2*abs(np.max(fit_y)),10,15))

            popt_lim, pcov_lim = curve_fit(gaussian, fit_x, fit_y, p0 = p0, sigma = fit_sig,\
                                    bounds = bounds)

            yfit = gaussian(vaxis, *popt_lim)
            red_chisq = np.nansum((fit_y - yfit[np.where(line)])**2/fit_sig**2)/ \
                        (np.sum(np.isfinite(fit_y))/oversamp - 4)


            # Determine the limit. SIMPLE VERSION, JUST UP THE PEAK BY 2 SIGMA.
            # TO SOME APPROXIMATION THIS IS A 95% UPPER LIMIT.
            amp = popt_lim[0]+2.*rms
            limit_fit = gaussian(vaxis, amp, popt_lim[1],popt_lim[2])
            limit_chisq = np.nansum((fit_y - limit_fit[np.where(line)])**2/fit_sig**2)/ \
                        (np.sum(np.isfinite(fit_y))/oversamp - 4)

            # GET THE INTEGRATED INTENSITY FROM THE LIMIT
            limit_ii = np.nansum(limit_fit*vdelt)

            confidence = 0.95

            coefs = popt_lim
            e_coefs = np.sqrt(np.diag(pcov_lim))



        #----------------------------------------------------------------------------
        #  CLEAN UP, CHANGE CASES, LOOP
        #
        # Work out the details of the fit: the fwhm, integrated intensity, etc. Then
        # possibly change cases between limit, double-horn, and gaussian fits. Add a
        # quality assessment and revise the region definitions. Then loop (unless we
        # seem to be trapped).
        #----------------------------------------------------------------------------

        # Calculate the residuals
        resid = spec - yfit

        # The peak
        peak_fit = np.nanmax(yfit)

        # The FWHM
        if type_fit == "limit":
            fwhm = popt_lim[2]*2.354
        else:
            fwhm = popt[2]*2.354


        # The integrated intensity
        ii_fit = np.nansum(yfit*vdelt)

        vaxis_detect_ind = np.where(yfit>0.01*peak_fit)

        #User dependent!
        if galaxy in ["ngc0628"] and xmid == 9.5 and line_name in ["CO10"]:
            vaxis_detect_ind = np.where(np.logical_and(vaxis>-50, vaxis<150))
            
        ii = np.nansum(spec[vaxis_detect_ind]*vdelt)
        
        e_ii_prop = np.nan
        if peak_fit == 0:
            e_ii = np.sqrt(1.5*fwhm/vdelt*1.)*vdelt*rms
        else:
            ct = np.nansum(np.array(yfit>0.01*peak_fit,dtype = int))

            e_ii = np.sqrt(ct*1.)*vdelt*rms
            if type_fit == "gaussian":
                # Do Gaussian Error Propagation
                e_ii_prop = ii*np.sqrt((e_coefs[0]/coefs[0])**2\
                                      +(e_coefs[2]/coefs[2])**2)
        

        # Redefine the line and empty regions
        if peak_fit > 0:
            empty = np.array(abs(vaxis)>fwhm, dtype = int) | always_empty
        else:
            empty = np.array(yfit > 0.001*peak_fit, dtype = int) | always_empty
        line = np.array(empty == 0, dtype = int)

        rms = np.nanstd(spec[np.where(empty)])
        

        #note the old case
        old_type = type_fit

        
        #peak_snr = np.nanmax(spec[np.where(line)])/rms
        peak_snr = np.nanmax(yfit)/rms
        
        int_snr_fit = ii_fit/e_ii_prop
        int_snr = ii/e_ii


        if (peak_snr > peak_snr_good) and (int_snr > int_snr_good):
            quality = 2.
        elif (peak_snr < peak_snr_limit) or (int_snr < int_snr_limit):
            quality = 0
            type_fit = "limit"
        else:
            quality = 1
        if peak<0:
            quality = 0
            type_fit = "limit"
        # Change cases

        if fwhm >= broad_line_thresh and quality == 2:

            type_fit = "doublehorn"



    #---------------------------------------------------------------------------
    # Return output
    #---------------------------------------------------------------------------
    #print(ii)
    #print(ct)
    #print(vdelt)
    #print(rms)
    #print(e_ii)
    #
    #print(vaxis[np.where(empty)])
    #print(spec[np.where(empty)])
    #
    #print(vaxis)
    #print(spec)
    #print("--------------")
    #plt.plot(vaxis, spec)
    #plt.plot(vaxis[np.where(empty)], spec[np.where(empty)])
    #plt.show()

    result_fit["yfit"] = yfit
    result_fit["spec_detec"] = np.nan*vaxis
    result_fit["spec_detec"][vaxis_detect_ind] = spec[vaxis_detect_ind]
    result_fit["resid"] = resid
    result_fit["type"] = type_fit
    result_fit["ii"] = ii
    result_fit["e_ii"] = e_ii
    result_fit["e_ii_prop"] = e_ii_prop
    result_fit["fwhm"] = fwhm
    result_fit["rms"] = rms
    result_fit["quality"] = quality
    result_fit["red_chisq"] = red_chisq
    result_fit["zero"] = zero
    result_fit["peak"] = peak
    result_fit["coefs"] = coefs
    result_fit["limit_ii"] = limit_ii
    result_fit["limit_fit"] = limit_fit
    result_fit["e_coefs"] = e_coefs
    return result_fit
