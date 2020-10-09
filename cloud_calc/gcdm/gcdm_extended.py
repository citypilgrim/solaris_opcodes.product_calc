# imports
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as sfft
import scipy.interpolate as sinp
import scipy.integrate as sint
import scipy.signal as sig

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *


# params
_pool = mp.Pool(processes=GCDMPROCNUM)


# main func
@verbose
@announcer
def main(
        nrbdic,
        combpolboo=True,
        plotboo=False,
):
    '''
    Extended gradient-based cloud detection. Using filters to smoothen out the
    signal to pass into GCDM algorithm

    Parameters
        nrbdic (dict): output from .nrb_calc.py
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    '''
    # computing gcdm mask
    gcdm_trm = r_trm

    # determining working array
    work_ra = SNR_tra
    workz_ra = z_tra

    # applying filter

    ## handling invalid values
    work_ra[work_ra < 0] = 0
    work_ra = np.nan_to_num(work_ra)

    ## low pass filter



if __name__ == '__main__':
    from ...nrb_calc import main as nrb_calc
    from ....file_readwrite import smmpl_reader

    '''
    Good profiles to observe

    20200922

    136
    269                         # very low cloud has to be handled with GCDM
    306
    375                         # double peaks
    423                         # double peak
    935                         # very sharp peak at low height, cannot see
    1030                        # noisy peak
    '''

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        # '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200805/202008050003.mpl',
        starttime=LOCTIMEFN('202009220000', UTCINFO),
        endtime=LOCTIMEFN('202009230000', UTCINFO),
        genboo=True,
    )

    main(nrb_d, combpolboo=True, plotboo=True)
