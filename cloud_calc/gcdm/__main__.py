# imports
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
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
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection
    Calculates cloud product up till defined SNR threshold; NOISEALTITUDE
    Scattering profile is kept in .constant_profiles

    Future
        - KEMPIRICAL might be passed in as an argument for optimization/quality
          check in the future

    Parameters
        nrbdic (dict): output from .nrb_calc.py
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    '''
    # reading data
    if combpolboo:
        NRB_tra = nrbdic['NRB_tra']
        SNR_tra = nrbdic['SNR_tra']
    else:
        NRB_tra = nrbdic['NRB2_tra']  # co-pol
        SNR_tra = nrbdic['SNR2_tra']
    r_trm = nrbdic['r_trm']

    # retrieving scattering profile
    try:                        # scanning lidar NRB
        setz_a = nrbdic['DeltNbintheta_a']
        setzind_ta = nrbdic['DeltNbinthetaind_ta']
        z_tra = nrbdic['z_tra']
    except KeyError:            # vertical lidar NRB
        setz_a = nrbdic['DeltNbin_a']
        setzind_ta = nrbdic['DeltNbinind_ta']
        z_tra = nrbdic['r_tra']

    # retreiving molecular profile
    rayleigh_aara = np.array([
        rayleigh_gen(*setz) for setz in setz_a
    ])
    rayleigh_tara = np.array([
        rayleigh_aara[setzind] for setzind in setzind_ta
    ])
    _, _, betamprime_tra, _, _, _ = [
        tra[0]
        for tra in np.hsplit(rayleigh_tara, rayleigh_tara.shape[1])
    ]

    # computing gcdm mask
    gcdm_trm = np.arange(r_trm.shape[1])\
        < (np.argmax((SNR_tra <= NOISEALTITUDE) * r_trm, axis=1)[:, None])
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

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200901/202009010500.mpl',
        # starttime=LOCTIMEFN('202009010000', UTCINFO),
        # endtime=LOCTIMEFN('202009010800', UTCINFO),
        genboo=True,
    )

    main(nrb_d, combpolboo=True, plotboo=True)
