# imports
from copy import deepcopy
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
from .integral_check import main as integral_check
from .mask_out import main as mask_out
from .noise_filter import main as noise_filter
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

# main func
@verbose
@announcer
def main(
        r_trm, z_tra, setz_a, setzind_ta,
        work_tra,
        plotboo=False, plotind=0
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
        work_tra (np.ndarray): working array
        plotboo (boolean): whether or not to plot computed results
        plotind (int): index in time axis to plot out
    Return
        gcdm_ta (np.ndarray): each timestamp contains a list of tuples for the clouds
    '''
    # working arrays being masked
    work_tra, z_tra, r_trm, setz_a = chunk_operate(
        work_tra, z_tra, r_trm, setz_a, setzind_ta, mask_out, procnum=GCDMPROCNUM
    )
    owork_tra = deepcopy(work_tra)
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

    # computing threshold using moving average
    work0_tra = np.nan_to_num(work_tra)
    work0_tra[work0_tra < 0] = 0  # handling invalid values
    barwork_tra = sig.convolve(
        work0_tra,
        np.ones((1, GCDMTHRESWINDOW)), mode='same'
    )/GCDMTHRESWINDOW
    ## accomodating for start of window
    work0_tra = np.nan_to_num(dzwork_tra)
    work0_tra[work0_tra < 0] = 0  # handling invalid values
    sbarwork_ta = np.mean(work0_tra[:, :GCDMTHRESWINDOW], axis=1)
    barwork_tra[:, :GCDMTHRESWINDOW] = sbarwork_ta[:, None]
    ## computing threshold
    amax_tra = np.concatenate([
        KEMPIRICALLOW * barwork_tra[:, :GCDMTHRESWINDOW],
        KEMPIRICALHIGH * barwork_tra[:, GCDMTHRESWINDOW:]
    ], axis=-1)
    amin_tra = np.concatenate([
        (1 - KEMPIRICALLOW) * barwork_tra[:, :GCDMTHRESWINDOW],
        (1 - KEMPIRICALHIGH) * barwork_tra[:, GCDMTHRESWINDOW:]
    ], axis=-1)

    # finding clouds applying algorithm on each axis
    pool = mp.Pool(processes=GCDMPROCNUM)
    gcdm_ta = np.array([
        pool.apply(
            gcdm_algo,
            args=(
                work_tra[i], dzwork_ra,
                z_tra[i], gcdm_trm[i],
                amin_tra[i], amax_tra[i])
        )
        for i, dzwork_ra in enumerate(dzwork_tra)
    ])
    pool.close()
    pool.join()

    d2zwork_tra, _, _, _ = chunk_operate(
        dzwork_tra, z_tra, r_trm, setz_a, setzind_ta,
        gradient_calc, procnum=GCDMPROCNUM
    )

    # performing check on gcdm_ta based on integral threshold
    pool = mp.Pool(processes=GCDMPROCNUM)
    gcdm_ta = np.array([
        pool.apply(
            integral_check,
            args=(
                owork_ra, work_tra[i],
                z_tra[i], gcdm_trm[i],
                gcdm_ta[i],
                GCDMINTEGRALTHRES
            ))
        for i, owork_ra in enumerate(owork_tra)
    ])
    pool.close()
    pool.join()


    # plot feature
    if plotboo:
        fig, (ax, ax1) = plt.subplots(ncols=2, sharey=True)
        yupperlim = z_tra.max()

        i = plotind

        # indexing commonly used arrays
        z_ra = z_tra[i]
        gcdm_rm = gcdm_trm[i]
        dzwork_ra = dzwork_tra[i][gcdm_rm]
        oz_ra = np.copy(z_ra)
        z_ra = z_ra[gcdm_rm]
        # amin, amax = amin_ta[i], amax_ta[i]
        amin_ra, amax_ra = amin_tra[i], amax_tra[i]

        # plotting zero derivative
        ax1.plot(owork_tra[i], oz_ra)
        ax1.plot(work_tra[i], oz_ra)

        # plotting first derivative
        dzwork_plot = ax.plot(dzwork_ra, z_ra)
        pltcolor = dzwork_plot[0].get_color()

        # plotting thresholds
        ax.plot(amax_ra, z_ra, color=pltcolor, linestyle='--')
        ax.plot(amin_ra, z_ra, color=pltcolor, linestyle='--')

        ## plotting clouds
        gcdm_a = gcdm_ta[i]
        for j, cld in enumerate(gcdm_a):
            if j >= len(_cloudmarker_l):
                j %= len(_cloudmarker_l)

            cldbot, cldpeak, cldtop = cld
            cldbotind = np.argmax(z_ra >= cldbot)
            ax.scatter(amax_ra[cldbotind], cldbot,
                       color=pltcolor, s=100,
                       marker=_cloudmarker_l[j], edgecolor='k')
            ax.scatter(0, cldpeak,
                       color=pltcolor, s=100,
                       marker=_cloudmarker_l[j], edgecolor='k')
            if not np.isnan(cldtop):
                cldtopind = np.argmax(z_ra >= cldtop)
                ax.scatter(amin_ra[cldtopind], cldtop,
                           color=pltcolor, s=100,
                           marker=_cloudmarker_l[j], edgecolor='k')

        plt.show()

    return gcdm_ta


if __name__ == '__main__':
    from ...nrb_calc import main as nrb_calc
    from ....file_readwrite import smmpl_reader
    from ...constant_profiles import rayleigh_gen

    '''
    Good profiles to observe

    20200922

    136
    269                         # very low cloud has to be handled with GCDM
    306
    375                         # double peaks
    423                         # double peak
    940                         # very sharp peak at low height, cannot see
    1032                        # noisy peak
    '''

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        # starttime=LOCTIMEFN('202010290000', UTCINFO),
        # endtime=LOCTIMEFN('202010291200', UTCINFO),
        # starttime=LOCTIMEFN('202009220000', UTCINFO),
        # endtime=LOCTIMEFN('202009230000', UTCINFO),
        starttime=LOCTIMEFN('202011141830', 8),
        endtime=LOCTIMEFN('202011141900', 8),
    )

    ts_ta = nrb_d['Timestamp']
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

    work_tra = NRB_tra/betamprime_tra
    # work_tra = SNR_tra

    plotind = 0
    # plotind = 940
    # plotind = 2500
    # plotind = 2000
    print(ts_ta[plotind])
    main(
        r_trm, z_tra, setz_a, setzind_ta,
        work_tra,
        plotboo=True, plotind=plotind
    )
