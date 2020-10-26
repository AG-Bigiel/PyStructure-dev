#!/usr/bin/python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import griddata

def toomreqgas(rotcurve_in,mom0_header,mom0_data,mom1_data,mom2_data,vref,distance,debug=False):
	""" create classical Toomre Q_gas radial profiles """
	""" from input moment maps and rotation curve """
	"""
	rotcurve_in:	Baraolo 3D output file, i.e. ascii format rotation curve with following columns:
			RAD(Kpc) RAD(arcs)  VROT(km/s) DISP(km/s) INC(deg)  P.A.(deg) Z0(pc)    Z0(arcs)  SIG(E20)  XPOS(pix) YPOS(pix) VSYS(km/s) VRAD(km/s)

	mom0_header:	moment 0 fits file header part

	mom0_data:	moment 0 fits file data part

	mom1_data:	moment 1 fits file data part

	mom2_data:	moment 2 fits file data part

	vref:		velocity in km/s used as reference (systemic velocity)

	distance:	distance in Mpc

        debug:          if True, some plots are shown as sanity check

	Note that moment maps must be aligned and must have same pixel scale.

	"""

	# get parameters from rotation curve
	file_rotcurve=rotcurve_in

	rotcurve_r=[]
	rotcurve_v=[]
	f = open(file_rotcurve)
	for line in f:
	    if line[0]<>'#':
        	l=line.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace(' ',',').split(',')
        	r=float(l[1])
        	vrot=float(l[2])
        	disp=float(l[3])
        	inclination=float(l[4])
        	PA=float(l[5])
        	x_center=int(float(l[9]))
        	y_center=int(float(l[10]))
        	rotcurve_r.append(r)
        	rotcurve_v.append(vrot)

	rotcurve_r=np.array(rotcurve_r,dtype=np.float64)
	rotcurve_v=np.array(rotcurve_v,dtype=np.float64)

	rotcurve_v /= np.sin(np.radians(inclination))

	# observed moment maps
	mh2=mom0_data
	velo=mom1_data
	vdisp=mom2_data

	############################################
	####### Convert to Surface Density #########
	cdelt=mom0_header['CDELT2']
	px_pc2 = (np.tan(np.radians(abs(cdelt)))*distance*1e6)**2
	mh2 /= px_pc2
	mh2 *= np.cos(np.radians(inclination))

	############################################
	########### Collapse radial ################
	# radial profile
	cen_x=x_center   #   same as rotcurve
	cen_y=y_center
	mh2_prof_y, vdisp_prof_y=radial_profile(mh2,velo,vdisp,vref,[cen_x,cen_y],inclination,PA,debug=debug)
	del mh2, vdisp
	mh2=mh2_prof_y
	vdisp=vdisp_prof_y

	########################################
	########### Vgal and Vdisp #############
	# convert velocities to m/s
	vdisp *= 1000.0


	########################################
	################# Rgal #################
	# convert to m
	rgal=np.array(range(len(mh2)))
	rgal_pc=np.tan(np.radians(cdelt*rgal))*distance*10**6
	rgal_pc *= u.pc
	rgal_m = rgal_pc.to(u.m).value

	########################################
	############### beta  ##################
	# beta = dlog(vgal)/dlog(rgal)

	# convert rotcurve_r from arcsec to pc
	rc_pc = np.tan(np.radians(rotcurve_r/3600.0))*distance*10**6
	rc_pc *= u.pc
	rc_m = rc_pc.to(u.m).value

	# convert rotcurve_v from km/s to m/s
	rc_ms = rotcurve_v*1000.0

	# interpolate rotcurve on gridpoints
	gridpoints=200
	rc_x=np.linspace(rc_m.min(),rc_m.max(),gridpoints)
	rc_y=griddata(rc_m,rc_ms,rc_x,method='cubic')

	# smooth rotcurve using moving average
	window=gridpoints/10
	avg_mask = np.ones(window) / window
	rc_y_avg = np.convolve(rc_y, avg_mask, 'same')

	# log rotcurve on interpolated gridpoints
	logvgal=np.log10(rc_y_avg)
	logrgal=np.log10(rc_x)

	# calculate beta for rotcurve
	beta=np.gradient(logvgal)/np.gradient(logrgal)

	###################
	##### PLOT RC #####
	###################
	this_x=np.concatenate((-rc_x*u.m.to(u.pc),rc_x*u.m.to(u.pc)))
	this_y=np.concatenate((-rc_y/1e3,rc_y/1e3))
	arr1inds = this_x.argsort()
	sorted_x = this_x[arr1inds]
	sorted_y = this_y[arr1inds]
	plt.plot(sorted_x,sorted_y,'r',ls='-',lw='3')
	plt.xlabel('$\mathsf{r\ [pc]}$')
	plt.ylabel('$\mathsf{v_{rot}\ [km/s]}$')
	plt.show()


	########################################
	############## Kappa  ##################
	kappa=np.full(beta.shape,np.nan)
	kappa[beta>-1] = 1.41 * (rc_y[beta>-1]/rc_x[beta>-1]) * (1.0+beta[beta>-1])**0.5
	kappa[beta<=-1]= 1.41 * (rc_y[beta<=-1]/rc_x[beta<=-1])


	########################################
	########### Toomre Q_gas ###############
	########## radial profile ##############
	########################################
	Q=np.full(mh2.shape,np.nan)
	kappa_arr=np.full(mh2.shape,np.nan)

	for i in range(len(mh2)):
	  radius=rgal_m[i]  # distance of pixel from center in meter, i.e. radius
          
	  if radius<=rc_m.max():
	    gas_sd=mh2[i]
	    gas_sd*=u.solMass*u.pc**(-2)
	    gas_sd=gas_sd.to(u.kg * u.m**-2)    # gas surface density in kg/m2
	    sigma=vdisp[i]    # velocity dispersion in m/s

	    # lookup kappa
	    closest_index=(np.abs(radius - rc_x)).argmin()
	    this_kappa=kappa[closest_index]

	    this_Q=(sigma*this_kappa) / (np.pi * const.G * gas_sd)

	    Q[i]=this_Q.value
	    kappa_arr[i]=this_kappa



	###########################################
	######### PLOT Q Profile ##################
	###########################################

	plt.plot(rgal_pc,Q,'r-',lw=4,label='$\mathsf{Q_{gas}}$')
	plt.axhline(1,ls='--',lw=2)
	plt.xlabel('$\mathsf{r\ [pc]}$')
	plt.ylabel('$\mathsf{Q_{gas}}$')
	plt.show()
		









def radial_profile(mom0, mom1, mom2, vref, center, inclination, pa, debug=False):
    """ elliptical radial profiles """

    pa+=90
    y, x = np.indices((mom0.shape))

    # rotate coordinate system to align major axis along x axis
    x_rot =  (center[0]-x)*np.cos(np.radians(pa)) + (center[1]-y)*np.sin(np.radians(pa))
    y_rot = -(center[0]-x)*np.sin(np.radians(pa)) + (center[1]-y)*np.cos(np.radians(pa))

    if debug:
        #################################################
        # make some plots to check alignment of ellipses
        z=mom1-vref
        z[z<0]=0
        z[z>200]=200
        z=z/200

        plt.scatter(x,y,c=z,cmap='RdBu')
        plt.title('Original Grid')
        plt.show()

        plt.scatter(x_rot,y_rot,c=z,cmap='RdBu')
        plt.title('Aligned Grid: x should be along major axis')
        plt.show()
        ###################################################


    # calculate radius for each pixel
    a=1.0
    b=a*np.cos(np.radians(inclination))

    r = np.sqrt( x_rot**2/a**2 + y_rot**2/b**2 )
    r = r.astype(np.int)

    if debug:
        ##################################################
        # another sanity check plot showing the final rings
        plt.scatter(x_rot,y_rot,c=r,cmap='RdBu')
        plt.title('Ring profiles')
        plt.show()
        ##################################################

    r=r.ravel()
    data=mom0.ravel()
    data2=mom2.ravel()

    keep1 = ~np.isnan(data)
    keep2 = ~np.isnan(data2)

    keep=np.logical_and(keep1, keep2)

    tbin = np.bincount(r[keep], data[keep])
    nr = np.bincount(r[keep])

    tbin2 = np.bincount(r[keep], data2[keep])
    nr2 = np.bincount(r[keep])

    radialprofile = tbin/nr
    radialprofile2 = tbin2/nr2

    return [radialprofile,radialprofile2]




