# imports
from .cloud_calc import main as cloud_calc
from .product_geolocate import main as product_geolocate
from .nrb_calc import main as nrb_calc
from ..global_imports.solaris_opcodes import *


# main func
@verbose
@announcer
def main(
        lidarname, mplreader,
        mplfiledir=None,
        startime=None, endtime=None,
        timestep=None, rangestep=None,
        angularoffset=0,

        combpolboo=True,

        pixelsize=None, gridlen=None,
        latitude=None, longitude=None, elevation=None,

):
    '''
    For a given input time, will return a dictionary with all data concerning
    product computation.

    It would perform a grid sampling of the product to geolocate the data if the
    pixelsize is specified

    Parameters
        lidarname (str): directory name of lidar
        mplreader (func): either mpl_reader or smmpl_reader
        mplfiledir (str): mplfile to be processed if specified, date should be
                          None
        start/endtime (datetime like): approx start/end time of data of interest
        timestep (int): if specified, will return a time averaged version of the
                        original,
                        i.e. new timedelta = timedelta * timestep
        rangestep (int): if specified, will return a spatially resampled version
                         of the original,
                         i.e. new range bin size = range bin * rangestep
        angularoffset (float): [deg] angular offset of lidar zero from north

        combpolboo (boolean): boolean that decides whether to use both cross and co
                              polarisation or just co polarisation for product
                              computation

        pixelsize (float): [km] pixel size for data sampling
        gridlen (int): length of map grid in consideration for geolocation
        latitude (float): [deg] coords of lidar
        longitude (float): [deg] coords of lidar
        elevation (float): [m] height of lidar from MSL

    Return
        product_d (dict):
            PRODUCTTIMESTAMPKEY (array like): contains timestamps for grid
            PRODUCTARRAYSKEY (dict): containing everything in nrb_d
            PRODUCTLATKEY (np.ndarray): latitude coordinates
            PRODUCTLONGKEY (np.ndarray): longitudinal coordinates

            ...other product dictionary keys: each dictionary contains all product
                                              information, augmented with grid sampling
    '''
    product_d = {}

    # computing NRB
    nrb_d = nrb_calc(
        lidarname, mplreader,
        mplfiledir=None,
        starttime=None, endtime=None,
        timestep=None, rangestep=None,
        angularoffset=angularoffset,
    )


    # product computation

    ## cloud
    product_d[CLOUDKEY] = cloud_calc(
        nrb_d, combpolboo,
    )


    # geolocating products
    product_d[PRODUCTLATKEY], product_d[PRODUCTLONGKEY] = product_geolocate(
            product_d,
            pixelsize, gridlen,
            latitude, longitude, elevation
        )

    return product_d


# running
if __name__ == '__main__':
    pass
