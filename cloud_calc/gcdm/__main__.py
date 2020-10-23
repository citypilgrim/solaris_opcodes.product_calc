# imports
import matplotlib.pyplot as plt
import numpy as np

from .gcdm_extended import main as gcdm_extended
from .gcdm_original import main as gcdm_original
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
        nrbdic,
        combpolboo=True,
        plotboo=False,
):
    '''
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection
    Calculates cloud product up till defined SNR threshold; NOISEALTITUDE
    Scattering profile is kept in .constant_profiles

    Parameters
        nrbdic (dict): output from .nrb_calc.py
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    Return
        gcdm_ta (np.ndarray): each timestamp contains a list of tuples for the clouds
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
        setz_a = nrbdic['DeltNbinpadtheta_a']
        setzind_ta = nrbdic['DeltNbinpadthetaind_ta']
        z_tra = nrbdic['z_tra']
    except KeyError:            # vertical lidar NRB
        setz_a = nrbdic['DeltNbinpad_a']
        setzind_ta = nrbdic['DeltNbinpadind_ta']
        z_tra = nrbdic['r_tra']

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

    # getting products from extended GCDM
    gcdm_ta = gcdm_extended(
        r_trm, z_tra, setz_a, setzind_ta,
        SNR_tra,
        plotboo=False,
    )

    # processing the product


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
    return gcdm_ta



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
    )

    main(nrb_d, combpolboo=True, plotboo=True)
