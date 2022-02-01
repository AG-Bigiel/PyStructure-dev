import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize,SqrtStretch)
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from spectral_cube import SpectralCube
import radio_beam
from astropy.wcs import WCS
from scipy import signal
from scipy import ndimage
from astropy.convolution import convolve, convolve_fft
import copy
from gauss_conv import *



#-------------------------------------------------------------------------------
# Import DATA
#-------------------------------------------------------------------------------

data_path = '/vol/alcina/data1/jdenbrok/Proj_I_2019/data/ngc5194_co21.fits'

data, header = fits.getdata(data_path, header = True)
vspace = np.arange(0,header["NAXIS3"])*header["CDELT3"] + header["CRVAL3"]
wcs = WCS(header)

target_resol = 30


cube = SpectralCube(data=data, wcs = wcs)
cube.beam = radio_beam.Beam(major=header["BMAJ"]*3600*u.arcsec, minor=header["BMIN"]*3600*u.arcsec, pa=0*u.deg)
beam = radio_beam.Beam(major=target_resol*u.arcsec, minor=target_resol*u.arcsec, pa=0*u.deg)
new_cube = cube.convolve_to(beam)



data_out, hdr_out = conv_with_gauss(in_data= data, in_hdr = header,
                                              target_beam = 30*np.array([1,1,0]),
                                              quiet = True)

data_new = np.array(new_cube.unmasked_data[:,:,:])

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title("SpecCube")
plt.imshow(np.nansum(data_new, axis = 0))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1,2,2)
plt.imshow(np.nansum(data_out, axis = 0))
plt.title("after")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()

plt.show()
