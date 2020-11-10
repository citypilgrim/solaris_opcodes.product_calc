# imports
import numpy as np

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
            CLOUDMASKKEY (np.ndarray): (time,  max # cld layers, 2(cld bot, cld top))
    '''
    cloud_d = {}

    # performing gcdm computation
    gcdm_ta = gcdm(nrbdic, combpolboo, plotboo)

    # seperating output into nested array of layers
    gcdm_tl2a = prodmask2ara_convert(gcdm_ta)

    # performing cloud logic
    cld_tl2a = gcdm_tl2a

    # removing base on minimal layer thickness

    ## finding invalid layers
    cld_tlm = (np.diff(cld_tl2a).squeeze() <= CLOUDMINTHICK)
    cld_tl2m = np.stack([cld_tlm]*2, axis=2)
    cld_tl2a[cld_tl2m] = np.nan

    ## removing entirely invalid layers
    invalidlayer_tl2m = np.isnan(cld_tl2a)
    invalidlayer_lm = ~(invalidlayer_tl2m.all(axis=(0, 2)))
    cld_tl2a = cld_tl2a[:, invalidlayer_lm, :]

    cloud_d[CLOUDMASKKEY] = cld_tl2a

    return cloud_d


# testing
if __name__ == '__main__':
    main()
