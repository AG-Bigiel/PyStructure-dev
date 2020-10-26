#!/usr/bin/python

import numpy as np
from astropy.io import fits
from toomreQ import *

hdu_m0=fits.open("NGC_4254_mh2_const.fits")
hdu_m1=fits.open("NGC_4254_1mom.fits")
hdu_m2=fits.open("NGC_4254_2mom.fits")

hdr=hdu_m0[0].header
m0=hdu_m0[0].data
m1=hdu_m1[0].data
m2=hdu_m2[0].data

rc="ringlog.txt"

distance=14.4   # Mpc
vref=2300.0     # km/s

toomreqgas(rc,hdr,m0,m1,m2,vref,distance,debug=False)

exit()
