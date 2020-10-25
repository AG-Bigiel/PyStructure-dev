import numpy as np

def deconvolve_gauss(meas_maj,       # measured major axis
                     beam_maj,       # beam major axis
                     meas_min = None,       # measured minor axis
                     meas_pa = None,         # measured position angle
                     beam_min = None,       # beam minor axis
                     beam_pa = None,         # beam position angle
                    ):

    """
    :param meas_maj:
    :param meas_min:
    :param meas_pa:
    :
    WARNING!!! Currently this appears to have issues returning a full
    ;  range of position angle values. This needs more investigation.

    ;  ADAPTED FROM gaupar.for in MIRIAD via K. Sandstrom

    ;
    ;  Determine the parameters of a gaussian deconvolved with another
    ;  gaussian.
    ;
    ;  Input:
    ;    bmaj1,bmin1        Major and minor FWHM of the source..
    ;    bpa1               Position angle of 1st gaussian, in degrees.
    ;    bmaj2,bmin2        Major and minor FWHM of gaussian to deconvolve with.
    ;    bpa2               Position angle of 2nd gaussian, in degrees.
    ;  Output:
    ;    bmaj,bmin          Major and minor axes of resultant gaussian.
    ;    bpa                Position angle of the result, in radians.
    ;    fac                Always 1 (for future use ...).
    ;    ifail              Success status: 0   All OK.
    ;                                       1   Result is pretty close to a
    ;                                           point source.
    ;                                       2   Illegal result.

    Modifications: 17 Oct 2019, converted IDL to python
    """
    #-----------------------------------------------------------------------
    # DEFAULTS AND DEFINITIONS
    #-----------------------------------------------------------------------


    # NO MINOR AXIS BEAM FOR BEAM
    if beam_min is None:
        print("[INFO]\t Minor axis not supplied. Assuming round measurement.")
        meas_min = meas_maj
    # NO POSITION ANGLE - DEFAULT TO 0
    if meas_pa is None:
        print("[INFO]\t Position not supplied. Assuming measurement PA = 0.")
        meas_po = 0.

    if beam_pa is None:
        print("[INFO]\t Position not supplied. Assuming beam PA = 0.")
        beam_pa = 0.

    #-----------------------------------------------------------------------
    # Calculations
    #-----------------------------------------------------------------------

    # Convert to radians
    meas_theta = np.deg2rad(meas_pa)
    beam_theta = np.deg2rad(beam_pa)

    # math
    alpha = (meas_maj*np.cos(meas_theta))**2 + (meas_min*np.sin(meas_theta))**2 - \
            (beam_maj*np.cos(beam_theta))**2 - (beam_min*np.sin(beam_theta))**2

    beta = (meas_maj*np.sin(meas_theta))**2 + (meas_min*np.cos(meas_theta))**2 - \
           (beam_maj*np.sin(beam_theta))**2 - (beam_min*np.cos(beam_theta))**2

    gamma = 2*((meas_min**2-meas_maj**2)*np.sin(meas_theta)*np.cos(meas_theta) - \
               (beam_min**2-beam_maj**2)*np.sin(beam_theta)*np.cos(beam_theta))

    s = alpha + beta
    t = np.sqrt((alpha-beta)**2 + gamma**2)

    #; FIND THE SMALLEST RESOLUTION

    limit = min(meas_min, meas_maj, beam_maj, beam_min)
    limit = 0.1*limit*limit

    # Two Cases:

    if ((alpha < 0) or (beta <0) or (s<t)):
        #failure
        src_maj = 0
        src_min = 0
        src_pa = 0

        print("[WARNING]\t Illegal alpha, beta, or s value.")
        worked = False
        # close to a point source:
        if (0.5*(s-t)<limit) and (alpha > - limit) and (beta>-limit):
            point = True
        else:
            point = False
    else:
        src_maj = np.sqrt(0.5*(s+t))
        src_min = np.sqrt(0.5*(s-t))
        if (abs(gamma)+ abs(alpha-beta)) == 0:
            src_pa = 0
        else:
            src_pa = np.rad2deg(0.5*np.arctan(-gamma/(alpha-beta)))
        worked = True
        point = False

    return src_maj, src_min, src_pa, [worked, point]
