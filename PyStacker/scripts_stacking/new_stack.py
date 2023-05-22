import numpy as np



def new_stacking(lines,n_spec = 500):
    #NUMBER OF CHANNELS FOR SPECTRA


    new_structure = {
#                 LOCATION OF SAMPLING POINT
                  "dist_mpc": np.nan ,\
                  "posang_deg": np.nan ,\
                  "incl_deg": np.nan ,\
                  "beam_as": np.nan ,\
#                 SHUFFLED SPECTRA AND VELOCITY AXIS
                   'vaxis_kms': np.zeros(n_spec)*np.nan,\
#                 PARAMETERS FOR STACK_BY_X
                   'xtype': '',\
                   'xmin': np.nan,\
                   'xmid': np.nan,\
                   'xmax': np.nan,\
                   'mask': '',\
#                   FITTED PARAMETERS FOR SHUFFLED SPECTRA FOR CO
                   

    }
    for line in lines:
        new_structure['spec_K_'+line]= np.zeros(n_spec)*np.nan,
        new_structure['rms_K_'+ line] =  np.nan
        new_structure['center_kms_'+ line] =  np.nan
        new_structure['peak_K_'+ line] =  np.nan
        new_structure['ii_K_kms_'+ line] =  np.nan
        new_structure['uc_ii_K_kms_'+ line] =  np.nan
        new_structure['limit_K_kms_'+ line] =  np.nan

    return new_structure
