import numpy as np

def hex_grid(ctr_x, ctr_y, spacing, radec = False, r_limit = None, e_limit = None):

    #Give x and y spacing of the grid
    x_spacing = spacing
    y_spacing = spacing*(np.sin(np.deg2rad(60)))

    #Estimate the x & y extend
    if e_limit is None and not r_limit is None:
        scale = r_limit
    elif r_limit is None and not e_limit is None:
        scale = e_limit/2
    else:
        raise TypeError("Not correct input for hex_grid. Provide either r_limit or e_limit")
    half_ny = np.ceil(scale/y_spacing)
    half_nx = np.ceil(scale/x_spacing)+1

    #Make the grid

    x = np.outer(np.ones(2*int(half_ny)+1), np.arange(2*int(half_nx)+1))
    y = np.outer(np.arange(2*int(half_ny)+1),np.ones(2*int(half_nx)+1))

    x -= half_nx
    y -= half_ny

    # Figure out the Bottom Left Corner

    x *= x_spacing

    #dot is a way of making sure, that the arrays are multiplyed togethre correctly
    x += 0.5*x_spacing*(np.dot(abs(y)%2==1, 1))
    y *= y_spacing

    # Keep the subset that matches the requested conditions
    r = np.sqrt(x**2 + y**2)
    if not r_limit is None:
        keep = np.where(r<r_limit)
        keep_ct = len(keep)
    else:
        keep = np.where(np.logical_and(abs(x)<e_limit/2, abs(y),e_limit/2))
        keep_ct = len(keep)

    if keep_ct == 0:
        return np.nan, np.nan

    # Create output arrays
    yout = y[keep] + ctr_y

    if radec:
        xout = (x[keep]/np.cos(np.deg2rad(yout))+ctr_x)
    else:
        xout = (x[keep] + ctr_x)

    return xout, yout
