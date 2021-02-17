import copy
from astropy.io import fits
import numpy as np

def twod_head(hdul_header):
    """
    NAME:

    twod_head

    PURPOSE:

    silly little program to hack a header from XXX to 2 dimensions. useful for
    stuff like hastrom that require specifically two dimensions to function.

    CATEGORY:

    glorified string command

    CALLING SEQUENCE:

    hacked_header = twod_head(real_header)
    MODIFICATION HISTORY:

    written - 11 apr 08 leroy@mpia.de in idl
    converted to python - 16 oct 19 jdenbrok
    """

    # Copy the original
    header_copy = copy.copy(hdul_header)

    #How many axes?
    naxis = hdul_header["NAXIS"]

    # Now set the number of axes to two
    header_copy["NAXIS"] = 2

    #check if certain keyword is in header
    if 'WCSAXES' in header_copy:
        header_copy['WCSAXES'] = 2

        #delete additional axes
    if naxis > 2:
        header_copy['WCSAXES'] = 2
        for i in range(3,naxis+1):
            del header_copy["*{}*".format(int(i))]
    return header_copy
