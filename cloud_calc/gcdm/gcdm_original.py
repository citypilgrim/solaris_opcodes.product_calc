# imports
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
from .mask_out import main as mask_out
from ...nrb_calc import chunk_operate
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
        plotboo (boolean): whether or not to plot computed results
    '''
    # working arrays being masked
    oz_tra, or_trm, osetz_a = z_tra, r_trm, setz_a  # 'o' stands for original
    CRprime_tra = NRB_tra / betamprime_tra
    CRprime_tra, z_tra, r_trm, setz_a = chunk_operate(
        CRprime_tra, z_tra, r_trm, setz_a, setzind_ta, mask_out, procnum=GCDMPROCNUM
    )
    SNR_tra, _, _, _ = chunk_operate(
        SNR_tra, oz_tra, or_trm, osetz_a, setzind_ta, mask_out, procnum=GCDMPROCNUM
    )

    # computing gcdm mask
    # this mask takes into account the padding at the front, and only performs
    # the SNR mask for the array without padding
    gcdm_trm = np.arange(r_trm.shape[1])\
        <= np.array([
            np.argmax(SNR_tra[i][r_rm] <= NOISEALTITUDE) + np.argmax(r_rm)
            for i, r_rm in enumerate(r_trm)
        ])[:, None]
    gcdm_trm *= r_trm

    # computing first derivative; altitude and masks are not affected
    dzCRprime_tra, _, _, _ = chunk_operate(
        CRprime_tra, z_tra, r_trm, setz_a, setzind_ta,
        gradient_calc, procnum=GCDMPROCNUM
    )

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
            args=(dzCRprime_ra, z_tra[i], gcdm_trm[i], amin_ta[i], amax_ta[i])
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
            if i != 1032:
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
            print(gcdm_a)
            import sys; sys.exit(0)

            for j, cld in enumerate(gcdm_a):
                if j >= len(_cloudmarker_l):
                    j %= len(_cloudmarker_l)

                cldbot, cldtop = cld
                ax.scatter(amax, cldbot,
                           color=pltcolor, s=100,
                           marker=_cloudmarker_l[j], edgecolor='k')
                try:
                    ax.scatter(amin, cldtop,
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
    return gcdm_ta



if __name__ == '__main__':
    from ...constant_profiles import rayleigh_gen
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
    )

    NRB_tra = nrb_d['NRB_tra']
    SNR_tra = nrb_d['SNR_tra']
    r_trm = nrb_d['r_trm']
    setz_a = nrb_d['DeltNbinpadtheta_a']
    setzind_ta = nrb_d['DeltNbinpadthetaind_ta']
    z_tra = nrb_d['z_tra']

    # retreiving molecular profile
    rayleigh_aara = np.array([
        rayleigh_gen(*setz) for setz in setz_a
    ])
    rayleigh_tara = np.array([
        rayleigh_aara[setzind] for setzind in setzind_ta
    ])
    _, _, betamprime_tra, _ = [
        tra[:, 0, :]
        for tra in np.hsplit(rayleigh_tara, rayleigh_tara.shape[1])
    ]

    main(
        r_trm, z_tra, setz_a, setzind_ta,
        SNR_tra, NRB_tra,
        betamprime_tra,
        plotboo=True
    )
