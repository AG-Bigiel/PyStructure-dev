import numpy as np
from astropy.io import fits
import copy

from polar_coord_funcs import *
from deconvolve_gaussian import *
from gauss_PSF import *
import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft
from astropy.utils.console import ProgressBar

def round_sig(x, sig=2):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def convolve_func(data, kernel, method = "direct"):
    """
    Perform either the direct or the fast fourier transform convolution
    """
    if method == "direct":
        conv_data = convolve(data,kernel)
    else:
        conv_data = convolve_fft(data,kernel)
        
    return conv_data
    
    
def conv_with_gauss(in_data,
                    in_hdr = None,
                    start_beam = None,
                    pix_deg = None,
                    target_beam = None,
                    no_ft = False,
                    in_weight = None,
                    out_weight_file= None,
                    out_file = None,
                    unc = False,
                    perbeam = False,
                    quiet = False):
    """
    NAME:
    ;
    ; conv_with_gauss
    ;
    ; PURPOSE:
    ;
    ; Wrapper to "convolve" (FFT) or "convol" (direct kernel
    ; multiplication) that will generate an elliptical gaussian PSF and
    ; then convolve it with the target image. Works on cubes and has the
    ; ability to handle not-a-number values.
    ;
    ; CATEGORY:
    ;
    ; Image manipulation.
    ;
    ; CALLING SEQUENCE:
    ;
    ; conv_with_gauss(in_data=data
    ;                 in_hdr=hdr
    ;                 start_beam=start_beam,
    ;                 pix_deg=pix_deg,
    ;                 target_beam=target_beam,
    ;                 out_file=out_file,
    ;                 in_weight = weight,
    ;                 no_ft=no_ft,
    ;                 uncertainty = unc,
    ;                 perbeam = perbeam,
    ;                 quiet=quiet)
    ;
    ;
    ; INPUTS:
    ;
    ; in_data: accepts either a cube (np.ndarray) or a FITS file name.
    ;
    ; in_hdr: an input header (not needed if a FITS file is supplied or if
    ; both a starting beam size and a pixel scale are supplied).
    ;
    ; target_beam: the target PSF measured in ARCSECONDS as a three
    ; element array in the form [major, minor, position angle] with
    ;    position angle assumed to measured north through east and east
    ; assumed to be low x values (standard astronomical conventions).
    ;
    ; OPTIONAL INPUTS:
    ;
    ; start_beam: the current FWHM of the image PSF, if this is not
    ; supplied then conv_with_gauss tries to look for the BMAJ and BMIN
    ; keywords in the header. Set this to 0 if you want to force
    ; convolution with the target_beam size.
    ;
    ; out_file: name of the output file (FITS). Only works if a header is
    ; also supplied, in which case the header is updated to reflect the
    ; new beam size.
    ; pix_deg: the pixel scale of the data in degrees. Useful in order to
    ; avoid having to pass a header (and so allows the program to be used
    ; as a general convolution routine).
    ;
    ; in_weight: perform a weighted convolution, multiplying the cube or map
    ; by the weight map before hand.
    ;
    ; unc: tell the program to treat the data as an "uncertainty" map and
    ; so square the data before convolution then scale the result by the
    ; sqrt of the increase in beam area. Appropriate for propogating
    ; uncertainties with convolution (e.g., convolve a noise cube).
    ;
    ; perbeam: apply a correction that handles units in the case of "per
    ; beam" units. This is almost always Jy/beam or MJy/beam. In this
    ; case, the program scales by the ratio of beam areas at the end to
    ; account for the new definition of the beam.
    ;
    ; KEYWORD PARAMETERS:
    ;
    ; quiet: if thrown, suppresses the report on flux conservation and
    ; kernel calculations.
    ;
    ; no_ft: passed directly to 'convolve.' Tells the program to use the astropy
    ; 'convol' function instead of a Fourier transform.
    ;
    ; OUTPUTS:
    ;
    ; a fits file written to 'out_file' if that parameter is specified and
    ; a header is supplied
    ;
    ; out_data: the convolved data as a variable
    ;
    ; out_hdr: the updated header (requires a header to be passed in)
    ;
    ; MODIFICATION HISTORY:
    ;
    ; documented - leroy@mpia.de 17 nov 08
    ; modified to do elliptical gaussians - karin 26 jun 11
    ; AKL - cleaned up and folded in to cprops. May have broken PA
    ;       convention.
    ;     - caught asymmetric kernel bug - dec 14
    ;
    ; Jakob den Brok:
    ;      - Oct 2020: Translated to IDL routine to Python
    ;      - Mar 2021: Cleaned and get rid of IDL syntax leftovers
    :
    ;-
    """

    #---------------------------------------------------------------
    # Read in or copy data
    #----------------------------------------------------------------

    if isinstance(in_data, str):
        # Data given in form of a string, need to load data
        data, hdr = fits.getdata(in_data, header = True)
    else:
        data = copy.deepcopy(in_data)
        if not in_hdr is None:
            hdr = in_hdr

    # Check if weights provided:
    if not in_weight is None:
        if isinstance(in_weight, str):
            # Data given in form of a string, need to load data
            weight, weight_hdr = fits.getdata(in_weight, header = True)
        else:
            weight = in_weight

        dim_wt = np.shape(weight)
        dim_data = np.shape(data)

        if dim_data[0]!=dim_wt[0] or dim_data[1]!=dim_wt[1]:
            print("[ERROR]\t Weight image must match data in X and Y size. Returning.")
            return


    if not target_beam is None:
        if isinstance(target_beam, list) or isinstance(target_beam, np.ndarray):
            if len(target_beam) == 1:
                target_beam = np.array([target_beam[0],target_beam[0], 0.0])
            elif len(target_beam) == 2:
                target_beam = np.array([target_beam[0],target_beam[1], 0.0])
        elif isinstance(target_beam, float) or isinstance(target_beam, int):
            target_beam = np.array([target_beam,target_beam, 0.0])


    if hdr is None:
        if pix_deg is None or start_beam is None:
            print("[ERROR]\t Requires a header OR a beam (start_beam=) and a plate scale (pix_deg=). Returning")
            return

    #---------------------------------------------------------------
    # PROCESS DATA BEFORE CONVOLUTION
    #----------------------------------------------------------------

    # Save total flux.
    flux_before = np.nansum(data)


    #; Measure the pixel size
    if pix_deg is None:
        pix_deg = get_pixel_scale(hdr)
    as_per_pix = pix_deg*3600.

    #; If we are treating the data as an uncertainty-type map (i.e., one
    #; that convolves as error propogation), then we need to square it
    #; before convolution. We will take the square root later.
    if unc:
        data = data**2

    #---------------------------------------------------------------
    # WORK OUT THE SIZE OF THE CONVOLUTION KERNEL AND BUILD IT
    #----------------------------------------------------------------

    # Identify the size of the beam that already described the data
    # set. Allow the user-supplied size (given by start_beam) to override
    # the header value.
    if not start_beam is None:
        current_beam = start_beam
        if isinstance(current_beam, float):
            current_beam = [current_beam, current_beam, 0.0]
        elif isinstance(current_beam, list) or isinstance(current_beam, np.ndarray):
            if len(current_beam) ==2:
                current_beam = [current_beam[0], current_beam[1], 0.0]
            else:
                print("[ERROR]\t Beam format not known, give either float or a list of BMIN and BMAJ")
                return
        else:
            print("[ERROR]\t Beam format not known, give either float or a list of BMIN and BMAJ")
            return


    else:
        bmaj = hdr["BMAJ"]*3600
        bmin = hdr["BMIN"]*3600
        bpa = hdr["BPA"]*3600
        current_beam = [bmaj, bmin, bpa]

    # Now "deconvolve" the starting beam from the target beam to figure
    # out what the convolution kernel should look like for this case.
    result_deconv  = deconvolve_gauss(target_beam[0], current_beam[0],
                                      target_beam[1],target_beam[2],
                                      current_beam[1],current_beam[2])


    kernel_bmaj = result_deconv[0]
    kernel_bmin = result_deconv[1]
    kernel_bpa = result_deconv[2]
    info_deconv = result_deconv[3]

    if info_deconv[0] == False:
        print("[ERROR]\t Cannot get kernel for this combination of beams. ")
        return
    if info_deconv[1]:
        print("[WARNING]\t The target and starting beam are very close.")

    # Figure out an appropriate size for the convolution kernel (forcing odd)

    minsize = 6.*np.rint(kernel_bmaj/as_per_pix) + 1.
    kern_size = int(minsize)
   
    # You get problems if the PSF is bigger than the image if you are
    # using the IDLAstro convolve. Set to the minimum dimension
    dim_data = np.shape(data)
    if len(dim_data)==3:
        dim_x = dim_data[1]
        dim_y = dim_data[2]
    else:
        dim_x = dim_data[0]
        dim_y = dim_data[1]
    
    if kern_size > dim_x or kern_size > dim_y:
        print("[Warning]\t PSF is very big compared to image.")
        kern_size = np.int(np.floor(min(dim_x,dim_y)/2-2)*2 + 1)

    # Build the PSF based on the kernel calculation. Note that the units
    # are pixels and the rotation associated with the position is taken to
    # be counterclockwise from up-down (hence the !pi/2). There's an
    # implict assumption that east is to the left in the image.


    kernel = gaussian_PSF_2D(kern_size ,
          [0., 1., kernel_bmaj/as_per_pix,
                kernel_bmin/as_per_pix, 0., 0., np.pi/2.+np.deg2rad(kernel_bpa)],
          True, True)

    #---------------------------------------------------------------
    # Do the convolution
    #----------------------------------------------------------------

    # Convolve with the PSF. If the data have three dimensions, do the
    # convolution for each plane. Otherwise convolve the image itself. In
    # order to speed up plane-by-plane convolution, we save the Fourier
    # transform of the PSF and turn off padding.
    if no_ft:
        method = "direct"
    else:
        method = "fft"



    if in_weight is None:
        #    This is the case where we do not weight the image. We just
        #    convolve with our normalized kernel.

        dim_data = np.shape(data)

        if len(dim_data) == 3:
            new_data = copy.deepcopy(data)
            print("[INFO]\t Start Cube Convolution...")
            for spec_n in ProgressBar(range(dim_data[0])):
                new_data[spec_n,:,:] = convolve_func(data[spec_n, :,:],kernel,method)
            print("[INFO]\t Done Cube Convolution.")
            data = new_data
        else:
            data = convolve_func(data, kernel,method)
    else:
        #    In this case we do weight the image. We convolve both the image
        #    and the weights, which can be 2d or 3d. Then we divide the final
        #    convolved image by the weights.

        dim_data = np.shape(data)
        dim_wt = np.shape(weight)
        if len(dim_data)==3:
            #   The weights can be an image or a cube. Take the appropriate
            #   case and multiply them by the data.
            if len(dim_wt)==2:
                weighted_data = copy.deepcopy(data)
                for spec_n in range(len(dim_data[0])):
                    weighted_data[spec_n,:,:] = data[sepc_n,:,:]*weight
            elif len(dim_wt) == 3:
                weighted_data = data*weight
            else:
                print("[ERROR]\t Weights can only be image or cube. Returning.")
                return

            #Convolve the weighted data to the new resolution plane by plane
            new_data = copy.deepcopy(weighted_data)
            for spec_n in range(dim_data[0]):
                new_data[spec_n,:,:] = convolve_func(weighted_data[spec_n, :,:],kernel,method)
            weighted_data = new_data

            #       Convolve the weights, taking into account their shape, and
            #       then divide the convolved weighted cube by the convolved
            #       weight cube.
            if len(dim_wt) == 2:
                weight = convolve_func(weight,kernel,method)
                    
                data = weighted_data
                for n_spec in range(dim_data[0]):
                    data[n_spec,:,:] = weighted_data[n_spec,:,:] / weight
            else:
                new_weight = copy.deepcopy(weight)
                for n_spec in range(dim_data[0]):
                    new_weight[n_spec,:,:] = convolve_func(weight[spec_n, :,:],kernel,method)
                weight = new_weight
                data = weighted_data / weight

        else:
                # The two-d only case

            weighted_data = weight*data
            weigted_data = convolve_func(weighted_data,kernel,method)
            weight = convolve_func(weight,kernel,  method)

            data = weighted_data / weight

        #;    So at the end "weight" holds the convolved weights and "data"
        #;    holds the result of the weighted convolution.


    #---------------------------------------------------------------
    # Clean up after the convolution
    #----------------------------------------------------------------
    #; In these two cases we will need to know the beam sizes before and
    #; after convolution.
    if unc or perbeam:
        if in_weight is None:
            print("[WARNING]\t Interaction of weighting with perbeam and unc is not clear.")
            print("[WARNING]\t Proceed at your own risk here.")

        #    Work out the pixels per beam at the start of the convolution
        if hasattr(current_beam, '__len__'):
            if len(current_beam) == 3:
                current_fwhm = np.sqrt(current_beam[0]*current_beam[1])
                ppbeam_start = ((current_fwhm / as_per_pix / 2)**2 / np.log(2)*np.pi)
            else:
                ppbeam_start = 1.
        else:
            ppbeam_start = 1.
        # Work out the pixels per beam at the end of the CONVOLUTION
        target_fwhm = np.sqrt(target_beam[0]*target_beam[1])
        ppbeam_final = ((target_fwhm / as_per_pix / 2)**2 / np.log(2)*np.pi)


    #; If the data were treated as uncertainty-like and so squared before
    #; convolution then we undo that here. In this case, we also scale by
    #; the square root of the amount of averaging. The idea here is to
    #; mimic the procedure of averaging independent noise measurement

    if unc:
        #;    Take the square root of the data.
        data = np.sqrt(data)

        #;    The scale factor is sqrt(N_before/N_after)
        scale_fac = np.sqrt(ppbeam_start/ppbeam_final)

        #;    Scale the data
        data *= scale_fac

    

    #; If we are requested to treat the units as "per beam" then adjust the
    #; final map by the ratio of beam areas (final beam/original beam).
    if perbeam:
        scale_fac = ppbeam_final / ppbeam_start

        data *= scale_fac



    #---------------------------------------------------------------
    # REPORT ON THE CALCULATION (UNLESS SUPPRESSED)
    #----------------------------------------------------------------i
    flux_after = np.nansum(data)
    flux_ratio = flux_after/flux_before
    if quiet == False:


        print("[INFO]\t Pixel Scale [as] = " + str(round_sig(as_per_pix,3)))
        print("[INFO]\t Starting beam [as] = " + str(current_beam))
        print("[INFO]\t Target FWHM [as] = " + str(target_beam))

        kernel_beam = [kernel_bmaj, kernel_bmin, kernel_bpa]

        print("[INFO]\t Convolution Kernel [as] = " + str(kernel_beam))
        if hasattr(kern_size,"len"):
            if len(kern_size) == 2:
                kern_size = kern_size[0]

        print("[INFO]\t PSF Grid Size [pix] = "+str(kern_size))
        print("[INFO]\t Flux Ratio = " + str(round_sig(flux_ratio)))

        unc_treated = "no"
        if unc:
            unc_treated = "yes"
        print("[INFO]\t Treated as uncertainty = "+unc_treated)

        beam_cor = "no"
        if perbeam:
            beam_cor = "yes"
        print("[INFO]\t Correctd per beam units = "+beam_cor)

    #---------------------------------------------------------------
    # Update the Header and write the output
    #----------------------------------------------------------------

    hdr['BMAJ'] = (target_beam[0]/3600., 'FWHM BEAM IN DEGREES')
    hdr['BMIN'] = (target_beam[1]/3600.,'FWHM BEAM IN DEGREES')
    hdr['BPA'] = (target_beam[2],'POSITION ANGLE IN DEGREES')

    hdr["HISTORY"] = 'Python CONV_WITH_GAUSS: convolved with '+\
                      str([kernel_bmaj, kernel_bmin, kernel_bpa]) + ' pixel gaussian'
    if unc:
        hdr["HISTORY"] = 'Python CONV_WITH_GAUSS: treated as an uncertainty map'

    if not out_file is None:
        fits.writeto(out_file,data, hdr)
    if not in_weight is None:
        if not out_weight_file is None:
            dim_wt = np.shape(weight)
            dim_data = np.shape(data)

            if len(dim_wt) == 2 and len(dim_data) == 2:
                fits.writeto(out_weight_file, weight, hdr)
            if len(dim_data) == 3 and len(dim_wt) == 2:
                fits.writeto(out_weight_file, weight, twod_head(hdr))
            if len(dim_data) == 3 and len(dim_wt) == 3:
                fits.writeto(out_weight_file, weight, hdr)


    return data, hdr






"""
#----------------------------------------
#Example on how to run the function
#----------------------------------------

#Load in the data
data, header = fits.getdata("/Users/jdenbrok/Desktop/PhD_Bonn/CLAWS/data/3D/original/ngc5194_co21_claws.fits", header = True)

# Specify the target resolution (in arcsec)
target_res_as = 33
data_out, hdr_out = conv_with_gauss(data, header, target_beam = target_res_as*np.array([1,1,0]))




int_map = np.nansum(data, axis = 0)
int_map_out = np.nansum(data_out, axis = 0)

plt.subplot(1,2,1)
plt.imshow(int_map, vmin = 0, vmax = 15, origin = "lower")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(int_map_out, vmin = 0, vmax = 15, origin = "lower")
plt.colorbar()

plt.show()
"""
