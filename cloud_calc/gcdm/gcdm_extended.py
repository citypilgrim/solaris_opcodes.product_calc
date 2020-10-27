# imports
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
from .mask_out import main as mask_out
from .noise_filter import main as noise_filter
from ...nrb_calc import chunk_operate
from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *


# params
_cloudmarker_l = [
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "v",
    "^",
    "<",
    ">",
    "X",
    "D",
    "d",
]

# main func
@verbose
@announcer
def main(
        r_trm, z_tra, setz_a, setzind_ta,
        SNR_tra,
        plotboo=False,
):
    '''
    Extended gradient-based cloud detection. Using filters to smoothen out the
    signal to pass into GCDM algorithm

    Parameters
        r_trm (np.ndarray): range mask
        z_tra (np.ndarray): altitude array
        setz_a (list): set of zipped descriptors for the different types of
                       altitude arrays in z_tra
        setzind_ta (np.ndarray): index of setz_a for the corresponding time axis
                                 in all 'tra' arrays
        SNR_tra (np.ndarray): Signal to noise
        plotboo (boolean): whether or not to plot computed results
    Return
        gcdm_ta (np.ndarray): each timestamp contains a list of tuples for the clouds
    '''
    # working arrays being masked
    SNR_tra, z_tra, r_trm, setz_a = chunk_operate(
        SNR_tra, z_tra, r_trm, setz_a, setzind_ta, mask_out, procnum=GCDMPROCNUM
    )
    work_tra = SNR_tra
    owork_tra = SNR_tra
    gcdm_trm = r_trm

    # filtering
    work_tra, _, _, _, = chunk_operate(
        work_tra, z_tra, r_trm, setz_a, setzind_ta,
        noise_filter,
        FILTERLOWPASSPOLY, FILTERLOWPASSCUT1, FILTERLOWPASSCUT2,
        FILTERSAVGOLWINDOW, FILTERSAVGOLPOLY,
        procnum=GCDMPROCNUM
    )

    # computing first derivative; altitude and masks are not affected
    dzwork_tra, _, _, _ = chunk_operate(
        work_tra, z_tra, r_trm, setz_a, setzind_ta,
        gradient_calc, procnum=GCDMPROCNUM
    )

    # computing thresholds
    work0_tra = np.copy(dzwork_tra).flatten()
    work0_tra[work0_tra < 0] = 0  # handling invalid values
    work0_tra[~(gcdm_trm.flatten())] = np.nan  # set nan to ignore in average
    work0_tra = work0_tra.reshape(*(gcdm_trm.shape))
    barwork_ta = np.nanmean(work0_tra, axis=1)
    amax_ta = KEMPIRICAL * barwork_ta
    amin_ta = (1 - KEMPIRICAL) * barwork_ta

    # finding clouds applying algorithm on each axis
    pool = mp.Pool(processes=GCDMPROCNUM)
    gcdm_ta = np.array([
        pool.apply(
            gcdm_algo,
            args=(dzwork_ra, z_tra[i], gcdm_trm[i], amin_ta[i], amax_ta[i])
        )
        for i, dzwork_ra in enumerate(dzwork_tra)
    ])
    pool.close()
    pool.join()


    # plot feature
    if plotboo:
        fig, (ax, ax1) = plt.subplots(ncols=2, sharey=True)
        yupperlim = z_tra.max()

        for i, z_ra in enumerate(z_tra):
            if i != 935:
                continue

            # indexing commonly used arrays
            gcdm_rm = gcdm_trm[i]
            amin, amax = amin_ta[i], amax_ta[i]
            dzwork_ra = dzwork_tra[i][gcdm_rm]
            oz_ra = np.copy(z_ra)
            z_ra = z_ra[gcdm_rm]

            # plotting first derivative
            dzwork_plot = ax.plot(dzwork_ra, z_ra)
            pltcolor = dzwork_plot[0].get_color()

            ## plotting thresholds
            ax.vlines([amin, amax], ymin=0, ymax=yupperlim,
                      color=pltcolor, linestyle='--')

            ## plotting clouds
            gcdm_a = gcdm_ta[i]
            for j, cld in enumerate(gcdm_a):
                if j >= len(_cloudmarker_l):
                    j %= len(_cloudmarker_l)

                cldbot, cldtop = cld
                ax.scatter(amax, cldbot,
                           color=pltcolor, s=100,
                           marker=_cloudmarker_l[j], edgecolor='k')
                ax.scatter(amin, cldtop,
                           color=pltcolor, s=100,
                           marker=_cloudmarker_l[j], edgecolor='k')

            # plotting zero derivative
            ax1.plot(owork_tra[i], oz_ra)
            ax1.plot(work_tra[i], oz_ra)

        # ax.set_ylim([0, 5])
        # ax1.set_xscale('log')
        plt.show()


    return gcdm_ta





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
    )

    NRB_tra = nrb_d['NRB_tra']
    SNR_tra = nrb_d['SNR_tra']
    r_trm = nrb_d['r_trm']
    setz_a = nrb_d['DeltNbinpadtheta_a']
    setzind_ta = nrb_d['DeltNbinpadthetaind_ta']
    z_tra = nrb_d['z_tra']

    main(
        r_trm, z_tra, setz_a, setzind_ta,
        SNR_tra,
        plotboo=True
    )
