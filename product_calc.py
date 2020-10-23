# imports
from .cloud_calc import main as cloud_calc
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

        smmplboo=True, pixelsize=None,
        combpolboo=True,

        pixelsize=True,
):
    '''
    For a given input time, will return
        1. the relevant working arrays for visualisation as
        2. dictionary containing the product mask

    each product mask is an array with the length of the timestamp array, each element
    in the array is a list containing tuples indicating the start and end of a product

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

        combpolboo (boolean): boolean that decides whether to use both cross and co
                              polarisation or just co polarisation for product
                              computation
    Return
        ret_d (dict): containing everything in nrb_d augmented with additonal working
                      arrays from the product calculation
        prodmask_d (dict): containing the product masks
        product_d (dict): containing useful product information computed from product
        long_a (np.ndarray): longitudinal coordinates
        lat_a (np.ndarray)L latitude coordinates
    '''
    # computing NRB
    nrb_d = nrb_calc(
        lidarname, mplreader,
        mplfiledir=None,
        starttime=None, endtime=None,
        timestep=None, rangestep=None,
    )

    # product computation
    prodmask_d = {}

    ## cloud
    prodmask_d['cloud'], product_d['cloud'] = cloud_calc(
        nrb_d, combpolboo, plotboo=False
    )


    # producing coordinates
    if smmplboo:
        '''product grid coordinates'''
    else:
        '''product static coordinates'''


    return nrb_d, prodmask_d, product_d, long_a, lat_a


# running
if __name__ == '__main__':
    '''
    The operational usage of this code is to print out the cloud bottoms for the
    corresponding grid.

    The user will specify a time input, and it would retreieve the latest product
    which would span the entire sky
    '''

    # retreiving optimal time
    '''need a function for this'''


    # running computation
    ret_d, prodmask_d, product_d, long_a, lat_a = main(
        LIDARNAME, smmpl_reader,
        startime, endtime,
        verbboo=False,

        smmplboo=True,
        combpolboo=True
    )

    # printing output, as specified by santo
    '''fill in code here'''
