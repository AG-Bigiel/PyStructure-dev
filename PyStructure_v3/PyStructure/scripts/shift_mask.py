import numpy as np
import astropy.units as u

def doppler(line, dict):
    """
    Calculate the radio Doppler shift for given frequencies.

    Parameters:
    line (str): The name of the spectral line for which the Doppler shift is calculated.
    dict (dict): A dictionary containing the spectral line data. The dictionary should have the following structure:
                 dict = {
                     'line_name': {
                         'freq': np.array([...]),  # Array of frequency values in GHz
                     },
                     ...
                 }

    Returns:
    int: Always returns 0. The Doppler shifts are stored in the 'doppler' key of the corresponding spectral line in the input dictionary.
    """
    nu_lines = dict[line]['freq'].copy()
    dopp = np.zeros(len(nu_lines)-1)
    for i in range(len(nu_lines)-1):
        temp1 = nu_lines[i+1]*u.GHz
        # calculate the dopplershift from the i-th to the 1st frequency
        temp2 = temp1.to(u.km/u.s, equivalencies=u.doppler_radio(nu_lines[0]*u.GHz))
        dopp[i] = temp2.value
    dict[line]['doppler'] = dopp
    return 0

def mask_shift(prior_mask, shift):
    """
    Apply a shift to the input mask.
    
    Parameters:
    prior_mask (np.ndarray): The input mask.
    shift (np.ndarray): The amount of pixels the mask is shifted.
    
    Returns:
    np.ndarray: The shifted mask.
    """
    max_shift = np.nanmax(abs(shift))
    mask_shifted = np.zeros((len(shift), len(prior_mask[:,0]), len(prior_mask[0,:])))
    mask_fine = np.zeros_like(prior_mask)
    # pad the mask so you don't roll the end of the mask back in at the beginning where it should still be 0
    prior_mask_pad = np.pad(prior_mask, [(0,), (max_shift,)])
    
    for i in range(len(shift)):
        roll = np.roll(prior_mask_pad, shift[i]) # shift the mask
        mask_shifted[i] = roll[:, max_shift:-max_shift] # get rid of the padding again

    for j in range(len(mask_shifted[:,0])):
        # combine the masks for all peaks
        mask_fine[mask_shifted[j]==1] = 1 
    
    return mask_fine