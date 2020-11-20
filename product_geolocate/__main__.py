# imports
import numpy as np

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
    array_d = product_d
    theta_ta = array_d['theta_ta']
    phi_ta = array_d['phi_ta']

    # centering on provided coordinates
    '''continue here, FIRST TESET OUT THE PYMAP3D PACKAGE'''

    for key in producttype_l:
        prodmask_tl3a = product_d[key][MASKKEY]

        # convert product to cartesian grid coordinates
        # (time*max no. of layers., 3(mask bottom, mask peak, mask top))
        xprodmask_tl3a = prodmask_tl3a \
            * np.tan(theta_ta)[:, None, None] * np.cos(phi_ta)[:, None, None]
        yprodmask_tl3a = prodmask_tl3a \
            * np.tan(theta_ta)[:, None, None] * np.sin(phi_ta)[:, None, None]

        ## flattening and removing all completemly invalid layers
        xprodmask_a3a = xprodmask_tl3a.reshape((-1, 3))
        yprodmask_a3a = yprodmask_tl3a.reshape((-1, 3))

        invalidlayer_am = ~(np.isnan(xprodmask_a3a).all(axis=-1))
        xprodmask_a3a = xprodmask_a3a[invalidlayer_am]
        yprodmask_a3a = yprodmask_a3a[invalidlayer_am]

        # locating product into it's respective pixel
        # each element in a list represents a pixel.
        # the order of elements in the list
        prodmaskind_l = []

        # finding product height distributions in each pixel to determine layers

        # averaging within the pixel

        # interpolating across pixels if there is an empty pixel

        # correcting product height for elevation of lidar


# testing
if __name__ == '__main__':
    main()
