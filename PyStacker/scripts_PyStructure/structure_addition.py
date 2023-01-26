import numpy as np
import pandas as pd
import copy



def add_band_to_struct(struct={}, band="", unit="", desc=""):
    """
    ; NAME:
    ;
    ; add_band_to_struct
    ;
    ; PURPOSE:
    ;
    ; Adds several standard fields associated with a band name to a
    ; structure, returning a new structure. Utility for building an
    ; intensity mapping database.
    ;
    ; USAGE
    ;
    ; new_struct = add_band_to_struct(struct=old_struct, band="bandname")
    ;
    ; NOTE:
    """

    tags = list(struct.keys())

    int_name = "INT_VAL_"+band.upper()
    uc_name = "INT_UC_"+band.upper()
    cov_name = "INT_COV_"+band.upper()
    res_name = "INT_RES_"+band.upper()
    unit_name = "INT_UNIT_"+band.upper()
    desc_name = "INT_DESC_"+band.upper()
    tpeak_name = "INT_TPEAK_"+band.upper()
    rms_name = "INT_RMS_"+band.upper()
    #additional moment maps
    mom1_name = "INT_MOM1_"+band.upper()
    emom1_name = "INT_EMOM1_"+band.upper()
    mom2_name = "INT_MOM2_"+band.upper()
    emom2_name = "INT_EMOM2_"+band.upper()
    ew_name = "INT_EW_"+band.upper()
    eew_name = "INT_EEW_"+band.upper()

    new_struct = copy.copy(struct)
    if int_name in tags:
        #... this is risky. Delete and replace instead?
        print("[WARNING]\t Band "+int_name+" already in structure.")
        print("[WARNING]\t Resetting values and returning.")

    new_struct[int_name] = np.nan
    new_struct[uc_name] = np.nan
    new_struct[cov_name] = np.nan
    new_struct[res_name] = np.nan
    new_struct[unit_name] = unit
    new_struct[desc_name]= desc
    new_struct[tpeak_name]  = np.nan
    new_struct[rms_name]  = np.nan
    new_struct[mom1_name]  = np.nan
    new_struct[emom1_name]  = np.nan
    new_struct[mom2_name]  = np.nan
    new_struct[emom2_name]  = np.nan
    new_struct[ew_name]  = np.nan
    new_struct[eew_name]  = np.nan

    return new_struct





def add_spec_to_struct (struct={}, line="", unit="", desc="", n_chan = 500):
    """
    NAME:

    add_band_to_struct

    PURPOSE:

    Adds several standard fields associated with a cube name to a
    structure, returning a new structure. Utility for building an
    spectroscopic mapping database.

    IDL is inflexible with regard to arrays in structures and backwards
    compatbility is an issue for us. As a stopgap, we will adopt the
    not-great approach of assuming that 500 elements is enough for most
    extragalactic cases. This number can be modified.

    USAGE

    new_struct = add_cube_to_struct(struct=old_struct, band="bandname")

    NOTE:
    -
    """
    tags = list(struct.keys())

    empty_spec = np.zeros(n_chan)*np.nan

    int_name = "SPEC_VAL_"+line.upper()
    vchan0_name = "SPEC_VCHAN0_"+line.upper()
    deltav_name = "SPEC_DELTAV_"+line.upper()
    uc_name = "SPEC_UC_"+line.upper()
    cov_name = "SPEC_COV_"+line.upper()
    res_name = "SPEC_RES_"+line.upper()
    unit_name = "SPEC_UNIT_"+line.upper()
    desc_name = "SPEC_DESC_"+line.upper()
    tpeak_name = "SPEC_TPEAK_"+line.upper()
    rms_name = "SPEC_RMS_"+line.upper()


    new_struct = copy.copy(struct)

    if int_name in tags:
        #... this is risky. Delete and replace instead?
        print("[WARNING]\t Cube "+int_name+" already in structure.")
        print("[WARNING]\t Resetting values and returning.")

    new_struct[int_name] = empty_spec
    new_struct[vchan0_name] = np.nan
    new_struct[deltav_name]  = np.nan
    new_struct[uc_name]  = np.nan
    new_struct[cov_name] = np.nan
    new_struct[res_name]  = np.nan
    new_struct[unit_name]  = unit
    new_struct[desc_name]  = desc
    new_struct[tpeak_name]  = np.nan
    new_struct[rms_name]  = np.nan

    return new_struct
