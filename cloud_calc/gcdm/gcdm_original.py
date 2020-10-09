# imports
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
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

_pool = mp.Pool(processes=GCDMPROCNUM)


# main func
@verbose
@announcer
def main(
        r_trm, z_tra, setz_a, setzind_ta,
        SNR_tra, NRB_tra,
        betamprime_tra,
        combpolboo=True,
        plotboo=False,
):
    '''
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection
    Calculates cloud product up till defined SNR threshold; NOISEALTITUDE
    Scattering profile is kept in .constant_profiles

    Parameters
        r_trm (np.ndarray): range mask
        z_tra (np.ndarray): altitude array
        setz_a (list): set of zipped descriptors for the different types of
                       altitude arrays in z_tra
        setzind_ta (np.ndarray): index of setz_a for the corresponding time axis
                                 in all 'tra' arrays
        SNR_tra (np.ndarray): Signal to noise
        NRB_tra (np.ndarray): normalised back scatter
        betamprime_tra (np.ndarray): attenuated backscatter for molecular profile
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    '''

    # computing gcdm mask
    gcdm_trm = np.arange(r_trm.shape[1])\
        <= np.array([
            np.argmax(SNR_tra[i][r_rm] <= NOISEALTITUDE) + np.argmax(r_rm)
            for i, r_rm in enumerate(r_trm)
        ])[:, None]
    gcdm_trm *= r_trm

    # computing first derivative
    CRprime_tra = NRB_tra / betamprime_tra
    dzCRprime_tra = gradient_calc(CRprime_tra, z_tra, setz_a, setzind_ta)

    # Computing threshold
    CRprime0_tra = np.copy(CRprime_tra).flatten()
    CRprime0_tra[~(gcdm_trm.flatten())] = np.nan  # set nan to ignore in average
    CRprime0_tra = CRprime0_tra.reshape(*(gcdm_trm.shape))
    barCRprime0_ta = np.nanmean(CRprime0_tra, axis=1)
    amax_ta = KEMPIRICAL * barCRprime0_ta
    amin_ta = (1 - KEMPIRICAL) * barCRprime0_ta

    # finding clouds; applying algorithm on each axis
    gcdm_ta = np.array([
        _pool.apply(
            gcdm_algo,
            args=(dzCRprime_ra, gcdm_trm[i], amin_ta[i], amax_ta[i])
        )
        for i, dzCRprime_ra in enumerate(dzCRprime_tra)
    ])
    _pool.close()
    _pool.join()


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
            dzCRprime_ra = dzCRprime_tra[i][gcdm_rm]
            oz_ra = np.copy(z_ra)
            z_ra = z_ra[gcdm_rm]

            # plotting first derivative
            dzCRprime_plot = ax.plot(dzCRprime_ra, z_ra)
            pltcolor = dzCRprime_plot[0].get_color()

            ## plotting thresholds
            ax.vlines([amin, amax], ymin=0, ymax=yupperlim,
                      color=pltcolor, linestyle='--')

            ## plotting clouds
            gcdm_a = gcdm_ta[i]
            for j, tup in enumerate(gcdm_a):
                if j >= len(_cloudmarker_l):
                    j %= len(_cloudmarker_l)

                cldbotind, cldtopind = tup
                ax.scatter(amax, z_ra[cldbotind],
                           color=pltcolor, s=100,
                           marker=_cloudmarker_l[j], edgecolor='k')
                try:
                    ax.scatter(amin, z_ra[cldtopind],
                               color=pltcolor, s=100,
                               marker=_cloudmarker_l[j], edgecolor='k')
                except IndexError:
                    # for cloud tops that cannot be found
                    # i.e. cldtopind == np.nan
                    pass


            # plotting zero derivative
            ax1.plot(CRprime_tra[i], oz_ra)
            # ax1.vlines([bmax_ta[i]], ymin=0, ymax=yupperlim,
            #            color=pltcolor, linestyle='--')

        ax.set_ylim([0, 5])
        ax1.set_xscale('log')
        plt.show()


    # returning
    '''got to think about this'''



if __name__ == '__main__':
    from ...nrb_calc import main as nrb_calc
    from ....file_readwrite import smmpl_reader

    '''
    Good profiles to observe

    20200922

    139
    183
    189
    269
    309
    380
    416
    944
    1032
    '''

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        # '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200805/202008050003.mpl',
        starttime=LOCTIMEFN('202009220000', UTCINFO),
        endtime=LOCTIMEFN('202009230000', UTCINFO),
        genboo=True,
    )

    main(nrb_d, combpolboo=True, plotboo=True)