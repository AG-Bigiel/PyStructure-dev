import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def make_axes(h,
              wcs= None,
              quiet = False,
              novec = False,
              vonly = False,
              simple = False):
    """
    ; NAME:
    ;   make_axes
    ;
    ; PURPOSE:
    ;
    ;   Makes axis arrays using 'exast' and 'xy2ad.' Returns units of decimal
    ;   degrees. Requires a .FITS header and gives the option of returning
    ;   coordinates for each point in the image.
    ;
    ; CALLING SEQUENCE:
    ;
    ;   make_axes, h $
    ;            , raxis=raxis, daxis=daxis $
    ;            , vaxis=vaxis $
    ;            , rimg=rimg, dimg=dimg $
    ;            , astrom = astrom $
    ;            , /quiet $
    ;            , /novec $
    ;            , /vonly $
    ;            , /simple
    ;
    ; INPUTS:
    ;
    ;   H - a single header structure (e.g. one returned by 'readfits')
    ;       KEYWORD PARAMATERS: none
    ;
    ; OUTPUTS
    ;
    ;   RAXIS - the right ascension axis pulled from the central row of
    ;           the image.
    ;   DAXIS - the declination axis pulled from the central column of the
    ;           image.
    ;   VAXIS - the velocity/z axis.
    ;   RIMG  - the right ascension of each point in the image.
    ;   DIMG  - the declination of each point in the image.
    ;
    ; MODIFICATION HISTORY:
    ;
    ;       Initial Documentation - Thu Nov 2
    ;                               Adam Leroy <aleroy@mars.berkeley.edu>
    ;       Got rid of silly use of 'cube' and allowed axis construction
    ;         using only the header. Still accepts cube as an argument,
    ;         but does not use it.
    ;                               AL, 7/30/03
    ;                               <aleroy@astro.berkeley.edu>
    ;       Renamed routine 'make_axes' and scrapped backwards
    ;       compatibility. No longer takes cube argument and now has
    ;       optional arguments to return the full RA and DEC images (2d
    ;       fields) in addition to the axes.
    ;                               AL, 7/30/03
    ;                               <aleroy@astro.berkeley.edu>
    ;
    ;       finally patched in the GLS/SFL thing.
    ;        - 1 Oct 2007 - leroy@mpia-hd.mpg.de
    ;
    ;       vectorized ?and sped things way way up? - 29 Nov 2007 leroy@mpia.de
    ;
    ;       added option to skip RA/DEC with /VONLY - 12 Dec 2008
    ;
    ;       converted to python: 21. Oct 2019 by J. den Brok

    """


    #; PULL THE IMAGE/CUBE SIZES FROM THE HEADER
    naxis = h['NAXIS']
    naxis1 = h['NAXIS1']
    naxis2 = h['NAXIS2']
    naxis3 = h['NAXIS3']

    # USE 'extast' TO EXTRACT A FITS ASTROMETRY STRUCTURE
    if wcs is None:
        astrom = WCS(h)
    else:
        astrom = wcs

    #; IF DATASET IS A CUBE THEN WE MAKE THE THIRD AXIS IN THE SIMPLEST WAY
    #; POSSIBLE (NO COMPLICATED ASTROMETRY WORRIES FOR FREQUENCY
    #; INFORMATION)

    if naxis >= 3:
        #;   GRAB THE RELEVANT INFORMATION FROM THE ASTROMETRY HEADER
        crpix = astrom.wcs.crpix
        cdelt = astrom.wcs.cdelt
        crval = astrom.wcs.crval

        #;   MAKE THE VELOCITY AXIS (WILL BE M/S)
        v = np.arange(naxis3)
        vdif = v - (h['CRPIX3']-1)
        vaxis = (vdif*h["CDELT3"]+h["CRVAL3"])

        if vonly:
            return vaxis


        #; IF 'SIMPLE' IS CALLED THEN DO THE REALLY TRIVIAL THING:
        if simple:
            print('[INFO]\t Using simple aproach to make axes.')
            print('[WARNING]\t BE SURE THIS IS WHAT YOU WANT! It probably is not')
            raxis = np.arange(naxis1)
            rdif = raxis - (h['CRPIX1']-1)
            raxis = (rdif * h['CDELT1']+ h['CRVAL1'])

            daxis = np.arange(naxis2)
            ddif = daxis - (h['CRPIX1']-1)
            daxis = (ddif * h['CDELT1']+ h['CRVAL1'])

            rimg = raxis # (fltarr(naxis2) + 1.)
            dimg = (np.zeros(naxis1) + 1.) # daxis
            return raxis, daxis, rimg, dimg


        # ToDo: More complicated stuff
