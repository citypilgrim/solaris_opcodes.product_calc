# imports
from .gcdm import main as gcdm
# from .ucdm import main as ucdm
from .prodmask2ara_convert import main as prodmask2ara_convert
from ...global_imports.solaris_opcodes import *


# main func
def main(
        nrbdic,
        combpolboo=True,
        plotboo=False,
):
    '''
    Produces cloud masks and the corresponding cloud information
    Currently ucdm is not in use.

    Parameters
        nrbdic (dict): output from .nrb_calc.py
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    Return
        cloud_d (dict):
            CLOUDMASKKEY (np.ndarray): each timestamp contains a list of tuples for the
                                       clouds
    '''
    cloud_d = {}

    # performing gcdm computation
    gcdm_ta = gcdm(nrbdic, combpolboo, plotboo)

    # seperating output into nested array of layers
    gcdm_tl2a = prodmask2ara_convert(gcdm_ta)

    # performing cloud logic
    cloud_d[CLOUDMASKKEY] = gcdm_tl2a
    '''remove thin cloud layers'''
    '''group near by clouds'''

    return cloud_d


# testing
if __name__ == '__main__':
    main()
