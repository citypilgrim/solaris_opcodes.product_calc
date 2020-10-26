# imports
from .gcdm import main as gcdm
# from .ucdm import main as ucdm
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

    # performing cloud logic
    '''update here after mplnet visualisation'''
    cloud_d[CLOUDMASKKEY] = gcdm_ta

    return cloud_d


# testing
if __name__ == '__main__':
    main()
