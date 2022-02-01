import numpy as np

def gaussian_PSF_2D(npix, a, center = False, normalize = False):
    """
    ; Create a two dimensional rotated Gaussian appropriate for use as a
    ; point spread function in convolution.
    ;
    ; my_gauss2d, npix, a, /center, /normalize, output=kernel
    ;
    ; Where:
    ;
    ; npix = number of pixels in the kernel (ideally odd), either n or [nx,ny]
    ;
    ; a = Gaussian shape parameters. Defined as an array where:
    ;
    ; a[0] = constant offset (zero level)
    ; a[1] = peak of Gaussian
    ; a[2] = full width at half max major axis
    ; a[3] = full width at half max minor axis
    ; a[4] = center in the x coordinate
    ; a[5] = center in the y coordinate
    ; a[6] = rotation in radians of x-axis CCW
    ;
    ; center = flag telling the program to center the image
    ;
    ; normalize = flag telling the program to normalize the PSF
    ;
    ; output = the output Gaussian array
    """

    valid_npix = False
    if isinstance(npix, int) or isinstance(npix, float):
        nx = npix
        ny = npix
        valid_npix = True
    if isinstance(npix, list) or isinstance(npix, np.ndarray):
        if len(npix)==2:
            nx = npix[0]
            ny = npix[1]
            valid_npix = True
    if valid_npix == False:
        print("[ERROR]\t Invalid npix array size")
        return

    xarr = np.reshape(np.tile(np.arange(nx),ny),(ny, nx))
    yarr = np.reshape(np.repeat(np.arange(ny), nx),(ny,nx))

    if center:
        cenx = (nx-1)/2
        ceny = (ny-1)/2
    else:
        cenx = a[4]
        ceny = a[5]

    fac = 2*np.sqrt(2*np.log(2))

    ang = a[6]
    const = a[0]
    peak = a[1]
    widthx = a[2]/fac
    widthy = a[3]/fac

    s = np.sin(ang)
    c = np.cos(ang)

    xarr = xarr - cenx
    yarr = yarr - ceny

    t = xarr * (c/widthx) + yarr * (s/widthx)
    yarr = xarr * (s/widthy) - yarr * (c/widthy)
    xarr = t

    u = np.exp(-0.5 * (xarr**2 + yarr**2))
    output = (const + peak * u)


    if normalize:
        tot = np.sum(output)
        output = output / tot


    return output
