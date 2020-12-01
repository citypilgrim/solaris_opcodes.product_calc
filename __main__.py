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
        starttime=None, endtime=None,
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
        mplfiledir (str): mplfile to be processed if specified, start/end time
                          do not have to be specified
        start/endtime (datetime like): approx start/end time of data of interest.
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
        nrb_d (dict):
            ..all keys can be found in .nrb_calc.__main__,
            ..might also include added keys for newly computed arrays

        product_d (dict):
            PRODUCTTIMESTAMPKEY (array like): contains timestamps for grid
            PRODUCTARRAYSKEY (dict): containing everything in nrb_d
            PRODUCTLATKEY (np.ndarray): latitude coordinates
            PRODUCTLONGKEY (np.ndarray): longitudinal coordinates

            ..other product dictionary keys: each dictionary contains all product
                                             information, augmented with grid sampling
    '''
    product_d = {}

    # computing NRB
    nrb_d = nrb_calc(
        lidarname, mplreader,
        mplfiledir,
        starttime, endtime,
        timestep, rangestep,
        angularoffset,
    )
    product_d[NRBKEY] = nrb_d

    # product computation

    ## cloud
    product_d[CLOUDKEY] = cloud_calc(
        nrb_d, combpolboo,
    )

    # geolocating products
    product_d = product_geolocate(
        pixelsize, gridlen,
        latitude, longitude, elevation,

        product_d,
        producttype_l=[
            CLOUDKEY
        ],

        peakonly_boo=False,
    )

    return product_d


# testing
if __name__ == '__main__':
    from ..file_readwrite import smmpl_reader

    main(
        'smmpl_E2', smmpl_reader,
        starttime=LOCTIMEFN('202011250000', 0),
        endtime=LOCTIMEFN('202011250430', 0),
        # endtime=LOCTIMEFN('202011251200', 0),
        angularoffset=140.6,

        pixelsize=5, gridlen=3,
        latitude=1.299119, longitude=103.771232,
        elevation=70,
    )
