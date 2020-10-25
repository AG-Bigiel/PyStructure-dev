import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.convolution import convolve, convolve_fft

def convolve(image, psf, noft = False, auto = False, correlate= False, no_pad = False):
    """
    ; NAME:
    ;       CONVOLVE
    ; PURPOSE:
    ;       Convolution of an image with a Point Spread Function (PSF)
    ; EXPLANATION:
    ;       The default is to compute the convolution using a product of
    ;       Fourier transforms (for speed).
    ;
    ;       The image is padded with zeros so that a large PSF does not
    ;       overlap one edge of the image with the opposite edge of the image.
    ;
    ;       This routine is now partially obsolete due to the introduction of  the
    ;       intrinsic CONVOL_FFT() function in IDL 8.1
    ;
    ; CALLING SEQUENCE:
    ;
    ;       imconv = convolve( image1, psf, FT_PSF = psf_FT )
    ;  or:
    ;       correl = convolve( image1, image2, /CORREL )
    ;  or:
    ;       correl = convolve( image, /AUTO )
    ;
    ; INPUTS:
    ;       image = 2-D array (matrix) to be convolved with psf
    ;       psf = the Point Spread Function, (size < or = to size of image).
    ;
    ;       The PSF *must* be symmetric about the point
    ;       FLOOR((n_elements-1)/2), where n_elements is the number of
    ;       elements in each dimension.  For Gaussian PSFs, the maximum
    ;       of the PSF must occur in this pixel (otherwise the convolution
    ;       will shift everything in the image).
    ;
    ; OPTIONAL INPUT KEYWORDS:
    ;
    ;       FT_PSF = passes out/in the Fourier transform of the PSF,
    ;               (so that it can be re-used the next time function is called).
    ;       FT_IMAGE = passes out/in the Fourier transform of image.
    ;
    ;       /CORRELATE uses the conjugate of the Fourier transform of PSF,
    ;               to compute the cross-correlation of image and PSF,
    ;               (equivalent to IDL function convol() with NO rotation of PSF)
    ;
    ;       /AUTO_CORR computes the auto-correlation function of image using FFT.
    ;
    ;       /NO_FT overrides the use of FFT, using IDL function convol() instead.
    ;               (then PSF is rotated by 180 degrees to give same result)
    ;
    ;       /NO_PAD - if set, then do not pad the image to avoid edge effects.
    ;               This will improve memory and speed of the computation at the
    ;               expense of edge effects.   This was the default method prior
    ;               to October 2009
    ; METHOD:
    ;       When using FFT, PSF is centered & expanded to size of image.
    ; HISTORY:
    ;       written, Frank Varosi, NASA/GSFC 1992.
    ;       Appropriate precision type for result depending on input image
    ;                               Markus Hundertmark February 2006
    ;       Fix the bug causing the recomputation of FFT(psf) and/or FFT(image)
    ;                               Sergey Koposov     December 2006
    ;       Fix the centering # BUG:
    ;                               Kyle Penner        October 2009
    ;       Add /No_PAD keyword for better speed and memory usage when edge effects
    ;            are not important.    W. Landsman      March 2010
    ;       Add warning when kernel type does not match integer array
    ;             W. Landsman Feb 2012
    ;       Don't force double precision output   W. Landsman July 2014
    """
    dim_im = np.shape(image)
    dim_psf = np.shape(psf)

    if len(dim_im) != 2 or noft:
        if auto:
            print("[ERROR]\t  Auto-correlation only for images with FFT. Returning")
            return image
        if correlate:
            return ndimage.convolve(image, psf)
        else:
            return ndimage.convolve(image, np.rot90(psf, k = 2))

    if no_pad:
