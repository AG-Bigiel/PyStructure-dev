import numpy as np



def new_stacking(lines,n_spec = 500):
    #NUMBER OF CHANNELS FOR SPECTRA


    new_structure = {
#                 LOCATION OF SAMPLING POINT
                  "gal": '' ,\
                  "tag": '' ,\
                  "ra_deg": np.nan ,\
                  "dec_deg": np.nan ,\
                  "dist_mpc": np.nan ,\
                  "posang_deg": np.nan ,\
                  "incl_deg": np.nan ,\
                  "beam_as": np.nan ,\
                  "rgal_as": np.nan ,\
                  "rgal_kpc": np.nan ,\
                  "rgal_r25": np.nan ,\
                  "theta_rad": np.nan ,\
#                 OBSERVED INTENSITIES: AVERAGE & SCATTER
#                  ; galex_fuv_Mjy_sr: nan ,\
#                  ; sc_galex_fuv_Mjy_sr: nan ,\
#                  ; galex_nuv_Mjy_sr: nan ,\
#                  ; sc_galex_nuv_Mjy_sr: nan ,\
#                  ; ha_Mjy_sr: nan ,\
#                  ; sc_ha_Mjy_sr: nan ,\
#                  ; mips24_Mjy_sr: nan ,\
#                  ; sc_mips24_Mjy_sr: nan ,\
#                  ; co_K_kms: nan ,\
#                  ; sc_co_K_kms: nan ,\
#                  ; hi_jybm_kms: nan ,\
#                  ; sc_hi_jybm_kms: nan ,\
#                  ; sc_sfr_sd: nan ,\
#                  ; hi_mom2_kms: nan ,\
#                  ; hi_mom1_kms: nan ,\
#                  ; co_mom1_kms: nan ,\
#                 SHUFFLED SPECTRA AND VELOCITY AXIS
                   'vfield_kms': np.nan,\
                   'veltype':'',\
                   'vaxis_kms': np.zeros(n_spec)*np.nan,\
                   'CO21_spec_K': np.zeros(n_spec)*np.nan,\
#                   hcn_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; th_co_spec_K: np.zeros(n_spec)*np.nan,\
#                   hcop_spec_K: np.zeros(n_spec)*np.nan,\
#                   c13o_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; hnc_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; co32_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; co10_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; h13cn_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; h13cop_spec_K: np.zeros(n_spec)*np.nan,\
#                  ; hn13c_spec_K: np.zeros(n_spec)*np.nan,\
#                   c18o_spec_K: np.zeros(n_spec)*np.nan,\
#                 PARAMETERS FOR STACK_BY_X
                   'xtype': '',\
                   'xmin': np.nan,\
                   'xmid': np.nan,\
                   'xmax': np.nan,\
                   'mask': '',\
                   'ncounts': np.nan,\
                   'narea_kpc2': np.nan,\
                   'ratio_usfpix': np.nan,\
                   'velcte_kms':np.nan,\
#                   FITTED PARAMETERS FOR SHUFFLED SPECTRA FOR CO
                   'ftype_CO21': '' , \
                   'fqual_CO21': np.nan , \
                   'rms_K_CO21': np.nan , \
                   'center_kms_CO21': np.nan , \
                   'peak_K_co21': np.nan , \
                   'fwhm_kms_fit_CO21': np.nan , \
                   'fwhm_kms_CO21':np.nan , \
                   'fwhm_lim_CO21': np.zeros((2,2))*np.nan , \
                   'ii_K_kms_CO21': np.nan , \
                   'uc_ii_K_kms_CO21': np.nan , \
                   'limit_K_kms_CO21': np.nan , \
                   #'fit_co21':np.zeros(n_spec)*np.nan , \
                   'coefs_cCO1':np.zeros(5)*np.nan , \
                   'e_coefs_CO21':np.zeros(5)*np.nan , \
                   'zero_offs_C=21':np.nan , \
                   'flag_CO21':np.nan , \

    }
    for line in lines:
        new_structure[line+'_spec_K']= np.zeros(n_spec)*np.nan,
        new_structure['ftype_'+ line] =  ''
        new_structure['fqual_'+ line] =  np.nan
        new_structure['rms_K_'+ line] =  np.nan
        new_structure['center_kms_'+ line] =  np.nan
        new_structure['peak_K_'+ line] =  np.nan
        new_structure['fwhm_kms_fit_'+ line] =  np.nan
        new_structure['fwhm_kms_'+ line] = np.nan
        new_structure['fwhm_lim_'+ line] =  np.zeros((2,2))*np.nan
        new_structure['ii_K_kms_'+ line] =  np.nan
        new_structure['uc_ii_K_kms_'+ line] =  np.nan
        new_structure['limit_K_kms_'+ line] =  np.nan
        #new_structure['fit_'+ line] = np.zeros(n_spec)*np.nan
        new_structure['coefs_'+ line] = np.zeros(5)*np.nan
        new_structure['e_coefs_'+ line] = np.zeros(5)*np.nan
        new_structure['zero_offs_'+ line] = np.nan
        new_structure['flag_'+ line] = np.nan


    return new_structure
