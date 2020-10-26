#!/usr/bin/python

import numpy as np

def calc_sigtir(i24,i70,i100,i160,i250):

    """    
    This is the Python version of "calc_sigir.pro" which
    is part of the "cpropstoo/physical" package
    translated to python by Johannes Puschnig, 2019

    NOTE: compared to the idl version, some flux combinations were added
    though still some combinations missing.

    Implements Table 3 of Galametz+ 13
    
    I've implemented the surface density conversions, the luminosity
    conversions are very similar, though not identical.
    """

    # CHECK WHICH BANDS WE HAVE
    have24 = i24>0 or False
    have70 = i70>0 or False
    have100 = i100>0 or False
    have160 = i160>0 or False
    have250 = i250>0 or False
    
    # CONSTANTS
    c = 2.99792458*1e10
    pc = 3.0857*1e18
    nu24 = c/(24.0*1e-4)
    nu70 = c/(70.0*1e-4)
    nu100 = c/(100.0*1e-4)
    nu160 = c/(160.0*1e-4)
    nu250 = c/(250.0*1e-4)
    
    # MJY/SR -> W/KPC^2
    fac24 = nu24*1e-17*1e-7*4*np.pi*(pc*1e3)**2
    fac70 = nu70*1e-17*1e-7*4*np.pi*(pc*1e3)**2
    fac100 = nu100*1e-17*1e-7*4*np.pi*(pc*1e3)**2
    fac160 = nu160*1e-17*1e-7*4*np.pi*(pc*1e3)**2
    fac250 = nu250*1e-17*1e-7*4*np.pi*(pc*1e3)**2
    
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # MONOCHROMATIC CONVERSIONS
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    
    # TBD
    
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # HYBRID CONVERSIONS
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    
    # 24+70
    if have24 and have70 and not have100 and not have160 and not have250:
        s_tir = \
        (3.925)*i24*fac24 + \
        (1.551)*i70*fac70
        return s_tir
    
    # 24 + 100
    elif have24 and have100 and not have70 and not have160 and not have250:
        s_tir = \
        (2.421)*i24*fac24 + \
        (1.410)*i100*fac100
        return s_tir
    
    # 24 + 160
    elif have24 and have160 and not have70 and not have100 and not have250:
        s_tir = \
        (3.854)*i24*fac24 + \
        (1.373)*i160*fac160
        return s_tir
    
    # 24 + 250
    elif have24 and have250 and not have70 and not have100 and not have160:
        s_tir = \
        (5.179)*i24*fac24 + \
        (3.196)*i250*fac250
        return s_tir
    
    # 70+100
    elif have70 and have100 and not have24 and not have160 and not have250:
        s_tir = \
        (0.458)*i70*fac70 + \
        (1.444)*i100*fac100
        return s_tir
    
    # 70+160
    
    # 70+250
    
    # 100+160
    
    # 100+250
    
    # 160+250
    
    # 24+70+100
    elif have24 and have70 and have100 and not have160 and not have250:
        s_tir = \
        (2.162)*i24*fac24 + \
        (0.185)*i70*fac70 + \
        (1.319)*i100*fac100
        return s_tir
    
    # 24+70+160
    elif have24 and have70 and have160 and not have100 and not have250:
        s_tir = \
        (2.126)*i24*fac24 + \
        (0.670)*i70*fac70 + \
        (1.134)*i160*fac160
        return s_tir

    # 24+70+250
    
    # 24+100+160
    elif have24 and have100 and have160 and not have70 and not have250:
        s_tir = \
        (2.708)*i24*fac24 + \
        (0.734)*i100*fac100 + \
        (0.739)*i160*fac160
        return s_tir
    
    # 24+100+250
    
    # 24+160+250
    
    # 70+100+160
    elif have70 and have100 and have160 and not have24 and not have250:
        s_tir = \
        (0.789)*i70*fac70 + \
        (0.387)*i100*fac100 + \
        (0.960)*i160*fac160
        return s_tir
    
    # 70+100+250
    
    # 70+160+250
    elif have70 and have160 and have250 and not have24 and not have100:
        s_tir = \
        (1.018)*i70*fac70 + \
        (1.068)*i160*fac160 + \
        (0.402)*i250*fac250
        return s_tir
    
    # 100+160+250
    
    # 24+70+100+160
    elif have24 and have70 and have100 and have160 and not have250:
        s_tir = \
        (2.051)*i24*fac24 + \
        (0.521)*i70*fac70 + \
        (0.294)*i100*fac100 + \
        (0.934)*i160*fac160
        return s_tir
    
    # 24+70+100+250
    
    # 24+70+160+250
    elif have24 and have70 and have160 and have250 and not have100:
        s_tir = \
        (2.119)*i24*fac24 + \
        (0.688)*i70*fac70 + \
        (0.995)*i160*fac160 + \
        (0.354)*i250*fac250
        return s_tir
    
    # 24+100+160+250
    elif have24 and have100 and have160 and have250 and not have70:
        s_tir = \
        (2.643)*i24*fac24 + \
        (0.836)*i100*fac100 + \
        (0.357)*i160*fac160 + \
        (0.791)*i250*fac250
        return s_tir
    
    # 70+100+160+250
    
    # 24+70+100+160+250
    elif have24 and have70 and have100 and have160 and have250:
        s_tir = \
        (2.013)*i24*fac24 + \
        (0.508)*i70*fac70 + \
        (0.393)*i100*fac100 + \
        (0.599)*i160*fac160 + \
        (0.680)*i250*fac250
        return s_tir
    
    else:
        print "CALC_SIGTIR Warning, SIGTIR could not be calculated."
        print i24,i70,i100,i160,i250
        return None
