# imports
import numpy as np
from pymap3d import ned2geodetic

from ...global_imports.solaris_opcodes import *


# main func
def main(
        pixelsize, gridlen,
        latitude, longitude, elevation,

        product_d,
        producttype_l,
):
    '''
    geolocates the products provided to a map like grid. It augments the data in
    product_d with the geolocated data.
    the product mask has to be of shape
    (time, no. of layers, 3(mask bottom, mask peak , mask top))

    Parameters
        pixelsize (float): [km] pixel size for data sampling
        gridlen (int): length of map grid in consideration for geolocation
        latitude (float): [deg] coords of lidar
        longitude (float): [deg] coords of lidar
        elevation (float): [m] height of lidar from MSL

        product_d (dict): dictionary containing all relevant information
        producttype_l (list): list of keys of the product types which we would to
                              perform pixel averaging and geolocation

    Return
        product_d (dict): with the following keys added
            LATITUDEKEY: (np.ndarray): latitude coordinates of pixel
            LONGITUDEKEY: (np.ndarray): longitude coordinates of pixel

    '''
    # reading product and relevant arrays
    array_d = product_d[NRBKEY]
    theta_ta = array_d['theta_ta']
    phi_ta = array_d['phi_ta']

    # centering on provided coordinates; finding center NED of pixels
    # we ignore elevation effects when translating between pixels
    centern, centere = 0, 0

    ## finding coordinate limits; [[left_lim, center, right_lim], ...]
    ## shape (gridlen, gridlen, 2(north, east), 3(left_lim, center, right_lim))
    if gridlen%2:
        gridrange = np.arange(
            -(gridlen//2)*pixelsize, (gridlen//2 + 1)*pixelsize, pixelsize
        )
    else:
        gridrange = np.arange(
            -(gridlen/2 - 0.5)*pixelsize, (gridlen/2 + 0.5)*pixelsize, pixelsize
        )
    coordlim_gg2a = np.stack(np.meshgrid(gridrange, gridrange), axis=-1)
    coordlim_gg23a = np.stack(
        [
            coordlim_gg2a - pixelsize/2,
            coordlim_gg2a,
            coordlim_gg2a + pixelsize/2
        ], axis=-1
    )
    coordlim_g23a = coordlim_gg23a.reshape((-1, 2, 3))

    for key in producttype_l:
        prodmask_tl3a = product_d[key][MASKKEY]

        # convert product to cartesian grid coordinates
        # (time*max no. of layers., 3(mask bottom, mask peak, mask top))
        xprodmask_tl3a = prodmask_tl3a \
            * np.tan(theta_ta)[:, None, None] * np.cos(phi_ta)[:, None, None]
        yprodmask_tl3a = prodmask_tl3a \
            * np.tan(theta_ta)[:, None, None] * np.sin(phi_ta)[:, None, None]

        ## flattening and removing all completemly invalid layers
        prodmask_a3a = prodmask_tl3a.reshape((-1, 3))
        xprodmask_a3a = xprodmask_tl3a.reshape((-1, 3))
        yprodmask_a3a = yprodmask_tl3a.reshape((-1, 3))

        invalidlayer_am = ~(np.isnan(xprodmask_a3a).all(axis=-1))
        prodmask_a3a = prodmask_a3a[invalidlayer_am]
        xprodmask_a3a = xprodmask_a3a[invalidlayer_am]
        yprodmask_a3a = yprodmask_a3a[invalidlayer_am]
        prodmask_a23a = np.stack([xprodmask_a3a, yprodmask_a3a], axis=1)

        # locating product into it's respective pixel
        ## using an array of masks of shape a3a, each element in the array is a pixel
        prodmask_ga23m = \
            (coordlim_g23a[:, None, :, [0]] <= prodmask_a23a[None, :, :])\
            * (coordlim_g23a[:, None, :, [2]] >= prodmask_a23a[None, :, :])
        prodmask_ga3m = prodmask_ga23m.prod(axis=2)

        ## boolean slicing arrays, one array for mask bottom, peak, and top
        prodbot_gam = prodmask_ga3m[..., 0]
        prodpeak_gam = prodmask_ga3m[..., 1]
        prodtop_gam = prodmask_ga3m[..., 2]
        prodbot_gAl, prodpeak_gAl, prodtop_gAl = [], [], []
        ### captical 'A' here represents variable length arrays in the list
        for i in range(gridlen**2):
            prodbot_gAl.append(prodmask_a3a[:, 0][prodbot_gam[i]])
            prodpeak_gAl.append(prodmask_a3a[:, 1][prodpeak_gam[i]])
            prodtop_gAl.append(prodmask_a3a[:, 2][prodtop_gam[i]])

        # finding product height distributions in each pixel to determine layers
        import matplotlib.pyplot as plt
        ind = 4
        plt.hist(
            [prodbot_gAl[ind], prodpeak_gAl[ind], prodtop_gAl[ind]]
        )

        hist, bin_edges = np.histogram(prodbot_gAl[ind])
        print(hist)
        print(bin_edges)

        plt.show()

        # averaging within the pixel

        # interpolating across pixels if there is an empty pixel

        # correcting product height for elevation of lidar

    return
