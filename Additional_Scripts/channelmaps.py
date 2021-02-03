import os,glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectral_cube import SpectralCube
import matplotlib as mpl
import matplotlib.pyplot as plt
import regions
from matplotlib.patches import Ellipse


'''
Author: Eibensteiner, C. + snippets from Github/stackoverflow
Purpose: creating channel maps (channels as they are, not integrated)
'''

# -------- changeable Parameters ------------
startchan   = -80      #starting channel
stepchan    = 10       #step 
region_list = regions.read_ds9('hcn.reg')  # subcube from ds9 region file

rows = 5 # number of rows of figure
cols = 4 # number of columns of figure
cmap = 'viridis'

# load cube
path        = ''        # path to fits file
file        = 'xy.fits' # fits file
cube_file   = path+file

cube        = SpectralCube.read(filename=cube_file,hdu=0)
sub_cube    = cube.subcube_from_regions(region_list)

#--------------------------------------

fig  = plt.figure(figsize=(9,9))

for i in range(20):
    print(i)

    # extraction of spectral slice
    ch_low = startchan*u.km/u.s + (i*stepchan)*u.km/u.s
    ch_high= startchan*u.km/u.s + (i*stepchan)*u.km/u.s

    this_chan = sub_cube.\
        with_spectral_unit(u.km/u.s).\
        spectral_slab(ch_low,ch_high)

    # get the spatial and velo axis as arrays
    velo, dec, ra = this_chan.world[:]

    # save the spatial size in variables
    sizex = this_chan.shape[2]
    sizey = this_chan.shape[1]

    # save channelmap in fits format
    this_chan.write('channel'+str(i)+'.fits',format='fits',overwrite=True)

    # read fits file
    hdu  = fits.open('channel'+str(i)+'.fits')
    data = hdu[0].data
    hdr  = hdu[0].header

    # update the header
    for hd in [hdr]:
        del hd['NAXIS']
        hd['WCSAXES']= 2
        hd['NAXIS']  = 2
        del hd['*3']
        #del hd['VELREF']
        #del hd['VELO-LSR']
        #del hd['SPECSYS']
        del hd['RESTFRQ']


    # update and save the header
    fits.writeto('channel'+str(i)+'.fits',data=data[0,:,:],header=hdr,overwrite=True)
    hdu.close()

    # read fits // changed header
    hdu  = fits.open('channel'+str(i)+'.fits')
    data = hdu[0].data
    hdr  = hdu[0].header

    # get pixscale
    cdelt2 = hdr['CDELT2']
    cdelt2 = abs(float(cdelt2)*3600.0)    # pixelscale in arcsec/pix

    # get beamsize
    targetbeam_h    = hdr['BMAJ']*3600.0
    targetbeam_w    = hdr['BMIN']*3600.0
    targetbeam_pa   = hdr['BPA']

    # get WCS
    wcs = WCS(hdr)


    #------------------------------
    # starting to plot
    #------------------------------

    ax  = fig.add_subplot(rows, cols, i+1, projection=wcs)
    im  = plt.imshow(data, interpolation='nearest',cmap=cmap,origin='lower')

    ra = ax.coords[0]
    de = ax.coords[1]
    ra.set_major_formatter('hh:mm:ss.s')
    de.set_major_formatter('dd:mm:ss')
    ra.set_ticks(spacing=60. * u.arcsec)#
    ra.set_axislabel('RA')
    de.set_axislabel('DEC')

    if i==0 or i==4 or i==8 or i==12:
        ax_ra=ax.coords[0]
        ax_de=ax.coords[1]
        ax_ra.set_major_formatter('dd:mm:ss')
        ax_de.set_major_formatter('dd:mm:ss')
        ax_ra.set_ticklabel_visible(False)
        ax_de.set_ticklabel_visible(True)
    elif i==16:
        ax_ra=ax.coords[0]
        ax_de=ax.coords[1]
        ax_ra.set_major_formatter('dd:mm:ss')
        ax_de.set_major_formatter('dd:mm:ss')
        ax_ra.set_ticklabel_visible(True)
        ax_de.set_ticklabel_visible(True)
    elif i>15:
        ax_ra=ax.coords[0]
        ax_de=ax.coords[1]
        ax_ra.set_major_formatter('dd:mm:ss')
        ax_de.set_major_formatter('dd:mm:ss')
        ax_ra.set_ticklabel_visible(True)
        ax_de.set_ticklabel_visible(False)
    else:
        ax_ra=ax.coords[0]
        ax_de=ax.coords[1]
        ax_ra.set_ticklabel_visible(False)
        ax_de.set_ticklabel_visible(False)

    # plot colorbar
    if i==3 or i==7 or i==11 or i==15 or i==19:
        cb = plt.colorbar(im, fraction=0.05,pad=0.0)
        cb.set_label(label='$\mathsf{K}$',weight='bold',fontsize=12)
        #'$\mathsf{K\ km\ s^{-1}}$'

    # plot the beam
    ell = Ellipse((+5,+5), targetbeam_w/cdelt2, targetbeam_h/cdelt2,targetbeam_pa,\
        color='black',transform=ax.get_transform('pixel'))
    ax.add_artist(ell)

    # plot number of channel / km/s
    info = str(startchan+(i*stepchan))
    plt.text(0.05,0.88,info,color='white',transform=ax.transAxes,fontsize=12,weight='normal',horizontalalignment='left')

plt.subplots_adjust(left=0.1,right=0.90,top=0.97,bottom=0.08,wspace=-0.05,hspace=0.05)
plt.savefig('co21-channelmap.pdf',dpi=300)
plt.show()
exit()
