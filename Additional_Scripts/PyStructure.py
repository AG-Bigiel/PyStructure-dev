"""
Class that helps to analyze the PyStructure (in .npy format)
"""
__author__ = "J. den Brok"


import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, FK5
import astropy.units as au
import calc_sigtir as calc
from astropy.wcs import WCS
from astropy.constants import L_sun
from astropy.io import fits
from trans_array_to_map import *


class PyStructure:
    #store the linenames

    def __init__(self, path):
        """
        :param path: String path to the .npy database
        """
        self.struct = np.load(path, allow_pickle = True).item()

        # initiate the linenames
        self.lines = []
        self.assign_lines()

        self.rgal = self.struct["rgal_kpc"]
        self.theta = self.struct["theta_rad"] + np.pi

    def assign_lines(self):
        #generate keyword that holds the linenames


        #iterate over the keys to find the linenames (lines that have shuffled spec)
        for key in self.struct.keys():
            if "SPEC_VAL_SHUFF" in key:
                self.lines.append(key.split("SHUFF")[-1])

    def get_coordinates(self, center = None):
        """
        :param center: the Ra, and Dec center coordinates. If provided, will return delta_ra, delta_dec,
                       othrwise, the absolute coordinates will be returned.
                       Need to be profided as string. E.g. "13:29:52.7 47:11:43"
        """

        ra = self.struct["ra_deg"]
        dec = self.struct["dec_deg"]

        if center is None:
            return ra, dec

        skycoords_ref = SkyCoord(center, frame=FK5, unit=(au.hourangle, au.deg))

        coords_map = (ra, dec) *au.deg
        skycoords_map = SkyCoord(ra=coords_map[0], dec = coords_map[1], frame=FK5)

        aframe = skycoords_ref.skyoffset_frame()
        delta_ra = skycoords_map.transform_to(aframe).lon.arcsec
        delta_dec = skycoords_map.transform_to(aframe).lat.arcsec

        return delta_ra, delta_dec

    def get_sigtir(self):
        i_70 = self.struct["INT_VAL_PACS70"]
        i_160 = self.struct["INT_VAL_PACS160"]
        i_250 = self.struct["INT_VAL_SPIRE250"]

        uc_70 = self.struct["INT_UC_PACS70"]
        uc_160 = self.struct["INT_UC_PACS160"]
        uc_250 = self.struct["INT_UC_SPIRE250"]
        s_tir, s_tir_uc = calc.calc_sigtir(i70 = i_70, i70_uc = uc_70,
                                           i160 = i_160, i160_uc = uc_160,
                                           i250 = i_250, i250_uc = uc_250)

        self.sigtir = s_tir
        return s_tir

    def get_sfr(self):
        self.get_sigtir()
        #Formula according to Murphy 2011
        self.sfr =  self.sigtir/L_sun.value*1.48e-10

        return self.sfr

    def quickplot_2Dmap(self,line, s = 50, cmap = None):
        """
        Plot a 2D map of a line that is provided

        optional parameters
        :param s: marker size

        """

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(1,1,1)
        ra = self.struct["ra_deg"]
        dec = self.struct["dec_deg"]

        if not cmap:
            cmap = "RdYlBu_r"
        im = ax.scatter(ra, dec, c = self.struct["INT_VAL_"+line], s=s, marker="h", cmap = cmap)
        ax.invert_xaxis()
        ax.set_ylabel("Decl.")
        ax.set_xlabel("R.A.")

        cb_ax = fig.add_axes([.91,.124,.04,.754])
        fig.colorbar(im,orientation='vertical',cax=cb_ax)

        plt.show()

    def get_vaxis(self, get_shuff = False):
        ref_line = self.lines[0]

        if get_shuff:
            try:
                vchan0 = self.struct["SPEC_VCHAN0_SHUFF"+ref_line]
                vdelt = self.struct["SPEC-DELTAV_SHUFF12CO32"+ref_line]
                naxis3 = len(self.struct["SPEC_VAL_SHUFF"+ref_line][0])
            except:
                return self.struct["SPEC_VAXISSHUFF"]
        else:
            try:
                vchan0 = self.struct["SPEC_VCHAN0_"+ref_line]
                vdelt = self.struct["SPEC_DELTAV_"+ref_line]
                naxis3 = len(self.struct["SPEC_VAL_"+ref_line][0])
            except:
                return self.struct["SPEC_VAXIS"]
        vaxis = np.arange(naxis3)*vdelt+vchan0

        return vaxis

    def get_ratio(self,line,sn= 5):
        """
        Compute the ratio of two lines
        :param line: list of two string: ["line1","line2"]
        Computes ratiio line1/line2
        """
        ratio = {}

        line1 = line[0]
        line2 = line[1]

        line1_ii = self.struct["INT_VAL_"+line1]
        line1_uc = self.struct["INT_UC_"+line1]

        line2_ii = self.struct["INT_VAL_"+line2]
        line2_uc = self.struct["INT_UC_"+line2]

        ratio_val = np.zeros_like(line1_ii)*np.nan
        ratio_uc = np.zeros_like(line1_ii)*np.nan
        ratio_ul = np.zeros_like(line1_ii)*np.nan
        ratio_ll = np.zeros_like(line1_ii)*np.nan

        id_det = np.array(np.logical_and(line1_ii/line1_uc>sn,line2_ii/line2_uc>sn),dtype=int)


        ratio_val[np.where(id_det)] = line1_ii[np.where(id_det)]/line2_ii[np.where(id_det)]
        ratio_uc[np.where(id_det)] = ratio_val[np.where(id_det)]*np.sqrt((line1_uc[np.where(id_det)]/line1_ii[np.where(id_det)])**2 + (line2_uc[np.where(id_det)]/line2_ii[np.where(id_det)])**2)

        ratio["ratio"] = ratio_val
        ratio["uc"] = ratio_uc

        id_ul = np.array(np.logical_and(line1_ii/line1_uc<sn,line2_ii/line2_uc>sn),dtype=int)
        ratio_ul[np.where(id_ul)] = 2/3*sn*line1_uc[np.where(id_ul)]/line2_ii[np.where(id_ul)]
        ratio["ulimit"] = ratio_ul

        id_ll = np.array(np.logical_and(line1_ii/line1_uc>sn,line2_ii/line2_uc<sn),dtype=int)
        ratio_ll[np.where(id_ll)] = line1_ii[np.where(id_ll)]/line2_uc[np.where(id_ll)]/(2/3*sn)
        ratio["llimit"] = ratio_ll

        return ratio
    def export_fits(self,data_array,fname,adjust_header=None, verbose=False):
        """
        Export a 2D or 3D PyStructure array to a FITS file.
        :param array: string or PyStructure array as np.array
        :param fname: file name of fits file used for writeto
        :param adjust_header: Dictionary with header keys and the corresponding value
        """
        if isinstance(data_array, str):
            if verbose:
                print("[INFO]\tInterpreting input as existing PyStructure key")
            data_array = self.struct[data_array]
        dims_data = np.shape(data_array)
        #make sure data data array has the same shape as either a 2D or 3D cube
        if len(dims_data)==1:
            if dims_data[0]!=len(self.struct['ra_deg']):
                print("[ERROR]\tInput data_array does not match dimensions of 2D map or 3D cube")
                return np.nan
        elif len(dims_data)==2:
            if dims_data[0]!=len(self.struct['ra_deg']) or dims_data[1]!=len(self.struct['SPEC_VAXIS']):
                print("[ERROR]\tInput data_array does not match dimensions of 2D map or 3D cube")
                return np.nan

        #step 1: regrid hexagonal to cartesian grid
        if len(dims_data)==1:
            gridspacing = self.struct['beam_as']/3
            datamap, newx, newy = array_to_map(self.struct["ra_deg"],self.struct["dec_deg"],data_array,gridspacing=gridspacing)

        #step 2: prepare the header
        hdu = fits.PrimaryHDU(data=datamap)

        # Create a WCS object
        wcs_header = fits.Header()
        wcs_header['CTYPE1'] = 'RA---SIN'
        wcs_header['CTYPE2'] = 'DEC--SIN'
        wcs_header['CRVAL1'] = np.mean(newx)
        wcs_header['CRVAL2'] = np.mean(newy)

        center_x = (newx.shape[1] + 1) / 2.0
        center_y = (newy.shape[0] + 1) / 2.0
        wcs_header['CRPIX1'] = center_x  # Assuming pixel indices start from 1
        wcs_header['CRPIX2'] = center_y  # Assuming pixel indices start from 1

        # Set CDELT1 and CDELT2 to the maximum pixel size
        pixel_scale = np.max(np.abs(np.diff(newx)))
        pixel_scale_str = "{:.2e}".format(pixel_scale)
        wcs_header['CDELT1'] = float(pixel_scale_str)
        wcs_header['CDELT2'] = float(pixel_scale_str)


        wcs_header['RADESYS'] = 'FK5'

        wcs = WCS(wcs_header)

        # Create a table with newx and newy
        table_data = fits.TableHDU.from_columns([
            fits.Column(name='NEWX', format='E', array=newx.flatten()),
            fits.Column(name='NEWY', format='E', array=newy.flatten())
        ])

        # Create a BinTableHDU from the Table object
        table_hdu = fits.BinTableHDU(table_data.data, header=table_data.header)

        # Create an HDU list and add both HDUs
        hdul = fits.HDUList([hdu, table_hdu])

        # Append the WCS header to the PrimaryHDU
        hdul[0].header.extend(wcs.to_header(), update=True)



        #step 3: save the
        hdul.writeto(fname, overwrite=True)
