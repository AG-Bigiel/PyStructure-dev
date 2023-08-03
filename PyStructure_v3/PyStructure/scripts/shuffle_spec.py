import numpy as np
import copy
from making_axes import *
import matplotlib.pyplot as plt


def shuffle(spec,
            vaxis,
            zero = None,
            new_vaxis = None,
            new_hdr = None,
            new_naxis = None,
            new_crval = None,
            new_crpix = None,
            new_cdelt = None,
            interp=None,
            missing=None,
            fft = False,
            quiet = False):



    """
    SHUFFLE
    (saved in cpropstoo-master/cubes/shuffle.pro)

    Uses interpolation to rearrange a spectrum, array of spectra, or
    spectral cube to sit on a new velocity grid. Optionally, specify a
    new zero point, either for each input spectrum or for each location
    in the cube.

    CALLING SEQUENCE

    new_spectrum = shuffle(spec = spec, vaxis = vaxis $
    , zero = zero, target_vaxis = new_vaxis, target_hdr = new_hdr $
    , target_crval = new_crval, target_crpix = new_crpix $
    , target_cdelt = new_cdelt, interp=interp $
    , missing=missing, fft = fft, quiet = quiet)

    REQUIRED INPUTS

    spec - the spectrum to regrid

    vaxis - the original velocity axis

    some combination of target_hdr, target_vaxis, and keywords to make a
    new velocity axis

    OPTIONAL INPUTS

    zero - a local or global offset applied to the velocity before regridding.

    interp - degree of interpolation

    missing - value of missing data. Generall NaN or maybe zero.

    """

    #------------------------------------------------------------------------
    # ERROR CHECKING AND INPUT PROCESSING
    #------------------------------------------------------------------------

    # Build the output velocity axis if only keywords (spacing, center,
    # etc.) and not a vector are supplied
    if new_vaxis is None:
        if not new_hdr is None:
            if quiet == False:
                print("[INFO]\t Extracting new velocity axis from header.")

            new_vaxis = make_axes(new_hdr, vonly = True)
        else:
            if quiet == False:
                print("[INFO]\t Attempting to build a new axis from keywords.")
            if new_cdelt is None:
                print("[INFO]\t ... defaulting to original channel width.")
                new_cdelt = vaxis[1] - vaxis[0]

            if new_crval is None or new_crpix is None:
                print("[INFO]\t ... defaulting to original reference value.")
                new_crval = vaxis[0]
                new_crpix = 1
            if new_naxis is None:
                print("[INFO]\t ... defaulting to original axis length.")
                new_naxis = len(vaxis)
            new_vaxis = (np.arange(new_naxis) - (new_crpix-1.))*new_cdelt + new_crval

    # If the new and old velocity axes are identical then throw an
    # informational error and return.
    if len(new_vaxis) == len(vaxis):
        if sum(new_vaxis != vaxis) == 0:
            if quiet == False:
                print("[INFO]\t  Your new and old velocity axes are identical.")
            return spec

    # Note the number of channels.

    n_chan = len(new_vaxis)

    # Record the size of the spectrum to figure out if it's an array of
    # spectra or cube with a plane of spectra.

    dim_spec = np.shape(spec)

    if len(dim_spec) == 2:
        shape = "ARRAY"
        n_spec = dim_spec[0]
    elif len(dim_spec)==3:
        shape = "CUBE"
        n_spec = dim_spec[1]*dim_spec[2]
    else:
        shape = "SPEC"
        n_spec = 1
    #  Default to no spectral shift. That is, the default behavior is a
    # simple regrid.
    if zero is None:
        if quiet == False:
            print("[INFO]\t Defaulting to regridding mode.")
        zero = 0.

    if isinstance(zero, int) or isinstance(zero, float) or hasattr(zero, "__len__"):
        #; Trap error in which zero is a mismatched size
        correct = True
        if hasattr(zero, "__len__"):
            if len(zero) != n_spec:
                correct = False
        if correct == False:
            print("[ERROR]\t The zero point vector should have either 1 or n_spec elements. Returning.")
            return np.nan
    # Note the number of channels in the old and new axis.
    orig_nchan = len(vaxis)
    orig_deltav = (vaxis[1] - vaxis[0])
    orig_chan = np.arange(orig_nchan)

    new_nchan = len(new_vaxis)
    new_deltav = (new_vaxis[1] - new_vaxis[0])
    new_chan = np.arange(new_nchan)

    # Default missing and interpolation
    if missing is None:
        missing = np.nan
    if interp is None:
        interp = 1
    valid_interp = [0,1,2]
    if not interp in valid_interp:
        if quiet == False:
            print("[WARNING]\t Invalid interpolation selection. Defaulting to linear.")
        interp = 1



    #--------------------------------------------------------------------------
    # Initialize the output
    #--------------------------------------------------------------------------
    if len(dim_spec) == 1:
        output = np.zeros(n_chan)*missing
    elif len(dim_spec) == 2:
        output = np.zeros((dim_spec[0],n_chan))*missing
    elif len(dim_spec) == 3:
        output = np.zeros((dim_spec[0], dim_spec[1], n_chan))*missing

    #------------------------------------------------------------------------
    # LOOP OVER SPECTRA AND DO THE INTERPOLATION
    #------------------------------------------------------------------------

    no_overlap_ct = 0

    for ii in range(n_spec):

        if quiet == False:
            pass
            #ToDo: some progress bar

        #    Retrieve the current spectrum into its own array
        if len(dim_spec)==3:
            yy = ii/dim_spec[0]
            xx = ii%dim_spec[0]
            this_spec = copy.copy(spec[xx,yy,:])
            if hasattr(zero, "__len__"):
                this_zero = zero[xx,yy]
            else:
                this_zero = zero
        else:
            if len(dim_spec)==2:
                this_spec = copy.copy(spec[ii,:])
            else:
                this_spec = copy.copy(spec)
            if hasattr(zero, "__len__"):
                this_zero = zero[ii]
            else:
                this_zero = zero
        # Recenter the current spectrum (this may be trivial for regridding)
        this_vaxis = vaxis - this_zero
        
        #check that both vaxis are increasing or decreasing. If one delta v increasing and other decressing, we flip the input spectrum and vaxis 
        if orig_deltav/new_deltav<0:
            this_vaxis = np.flip(this_vaxis)
            this_spec = np.flip(this_spec)
            
        # Check overlap of the recentered spectrum
        max_this_vaxis = max(this_vaxis)
        min_this_vaxis = min(this_vaxis)

        # Initialize a new spectrum and fill it with our "missing" value.
        new_spec = np.zeros(new_nchan)*missing

        # Find overlap
        channel_mapping = np.interp(new_vaxis, this_vaxis,orig_chan)


        overlap = np.where(np.logical_and(channel_mapping > 0.0, \
                     channel_mapping < orig_nchan-1))[0]
        overlap_ct = len(overlap)

        if overlap_ct == 0:
            no_overlap_ct +=1
            continue

        # Do the interpolation, with the method depending on the user
        # input.

        if interp == 0:
            #Nearest neighbor interpolation
            new_spec[overlap] = this_spec[np.array(np.rint(channel_mapping[overlap]), dtype = int)]
        elif interp == 1:
            new_spec[overlap] = np.interp(new_vaxis[overlap], this_vaxis,this_spec)
        else:
            print("[WARNING]\t Higher order interpolation not yet implemented.")
            new_spec[overlap] = np.interp(new_vaxis[overlap], this_vaxis,this_spec)



        #----------------------------------------------------------------------
        # Save the result in an output array.
        #----------------------------------------------------------------------


        if len(dim_spec) == 3:
            output[xx,yy,:] = new_spec
        elif len(dim_spec) == 2:
            output[ii,:] = new_spec
        else:
            output = new_spec

    return output


"""
# Test
spec = np.array([0.1,-0.08,0.03,0.0,-0.12,0.01,0.2,0.5,1,4,1,0.4,0.1,0.06,-0.1])
vaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
new_vaxis = np.array([1.5,2,2.5,3,3.5,4,4.5,5,5.5,6])

sh = shuffle(spec = spec, vaxis = vaxis, new_vaxis = new_vaxis)

import matplotlib.pyplot as plt

plt.plot(vaxis, spec)
plt.plot(new_vaxis, sh)
plt.show()
print(sh)
"""
