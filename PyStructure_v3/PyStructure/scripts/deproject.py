import numpy as np

def deproject(ra, dec, galpos, vector = False,
              gal=None):


    """
    ; NAME: deproject
    ;
    ; PURPOSE:
    ; Takes a center, position angle and inclination and computes
    ; deprojected radii and projected angle.
    ;
    ; CALLING SEQUENCE:
    ; deproject, ra, dec, galpos, RIMG = rimg, DIMG = dimg $
    ;          , RGRID = rgrid, TGRID = tgrid $
    ;          , XGRID = deproj_x, YGRID = deproj_y,GAL=gal
    ;
    ; INPUTS:
    ;
    ;   RA - the right ascenscion array corresponding to the x-axis. Can
    ;        also be an image if curvature is important.
    ;   DEC - the declination array corresponding to the y-axis. Can also
    ;         be an image if curvature is important.
    ;   GALPOS - the standard galaxy position/orientation array
    ;            [vlsr, pa, inc, xctr, yctr] (all in degrees & km/s)
    ;            or just [pa, inc, xctr, yctr] if 4 elements
    ;
    ; OUTPUTS:
    ;
    ;   RGRID - the grid of galactocentric radii
    ;   TGRID - the grid of angle from P.A.
    ;   XGRID - the grid of deprojected X-values
    ;   YGRID - the grid of deprojected Y-values
    ;
    ; KEYWORDS
    ;
    ;   VECTOR - tells DEPROJECT to return vectors matched in size to RA and DEC,
    ;            useful e.g. for radial profiles.
    ;
    ;
    ; MODIFICATION HISTORY:
    ;
    ;   -Written by
    ;      Adam Leroy < aleroy@mars.berkeley.edu > Thurs Jan 11, 2001
    ;   - Dramatically altered by
    ;      Adam Leroy < aleroy@mars.berkeley.edu> Thurs, Jan 18, 2001
    ;   - Hepped up on crack by
    ;      Adam Leroy < aleroy@mars.berkeley.edu> Mon, Feb 26, 2001
    ;   - Stripped down to its skivvies by
    ;      Adam Leroy < aleroy@mars.berkeley.edu> Wed, Mar 14, 2001
    ;   - Sign error tracked down and ground beneath the boot of
    ;      Adam Leroy < aleroy@mars.berkeley.edu> Mon, Apr 2, 2001
    ;   - Small concession to the curvature of the sky made by
    ;      Adam Leroy < aleroy@astro.berkeley.edu> Wed, Oct 22, 2003
    ;   - Added ability to work on vectors only (untested)
    ;      Adam Leroy < leroy@mpia-hd.mpg.de Mon, Oct 1, 2007
    ;   - Eats galaxy structures and takes 4 element galpo
    ;      Adam Leroy < leroy@mpia-hd.mpg.de Mon, Apr 7, 2008
    ;   - Converted to Python from IDL, Oct. 17, 2019
    ;      Jakob den Brok

    ; EXPAND THE GALAXY ORIENTATION VECTOR
    """
    np.seterr(divide='ignore', invalid='ignore')


    if not gal is None:
        pa = np.deg2rad(gal["posang_deg"])
        inc = np.deg2rad(gal["incl_def"])
        xctr = gal["ra_deg"]
        yctr = gal["dec_deg"]
    elif len(galpos) == 5:
        vlsr = galpos[0]
        pa   = np.deg2rad(galpos[1])
        inc  = np.deg2rad(galpos[2])
        xctr = galpos[3]
        yctr = galpos[4]

    else:
        pa   = np.deg2rad(galpos[0])
        inc  = np.deg2rad(galpos[1])
        xctr = galpos[2]
        yctr = galpos[3]

    ra_size = np.shape(ra)
    dec_size = np.shape(dec)

    if ra_size[0]==1 and vector==False:
        # IF THE USER HAS SUPPLIED ARRAYS AND NOT IMAGES THEN MAKE 2-D GRID,
        # ONE CONTAINING RA AND ONE CONTAINING DEC
        rimg = np.outer(ra, np.ones(len(dec)))
        dimg = np.outer(np.ones(len(ra)), dec)
    else:
        rimg = ra
        dimg = dec

    # RECAST THE RA AND DEC ARRAYS IN TERMS OF THE CENTERS
    # ARRAYS ARE NOW IN DEGREES FROM CENTER
    xgrid = (rimg - xctr)*np.cos(np.deg2rad(yctr))
    ygrid = (dimg - yctr)

    # ROTATION ANGLE (ROTATE YOUR X-AXIS UP TO THE MAJOR AXIS)
    rotang =  (-1.*(pa - 1.0*np.pi/2.))

    # MAKE 2-D GRIDS FOR ROTATED X AND Y
    deproj_x = xgrid * np.cos(rotang) + ygrid * np.sin(rotang)
    deproj_y = ygrid * np.cos(rotang) - xgrid * np.sin(rotang)

    # REMOVE INCLINATION EFFECT
    deproj_y = deproj_y / np.cos((inc))

    # MAKE GRID OF DEPROJECTED DISTANCE FROM THE CENTER
    rgrid = np.sqrt(deproj_x**2 + deproj_y**2)

    # MAKE GRID OF ANGLE W.R.T. PA
    tgrid = np.arctan2(deproj_y, deproj_x)

    return rgrid, tgrid
