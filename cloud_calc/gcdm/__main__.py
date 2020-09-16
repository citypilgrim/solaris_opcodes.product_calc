# imports
import matplotlib.pyplot as plt
import numpy as np

from .gcdm_algo import main as gcdm_algo
from .gradient_calc import main as gradient_calc
from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *


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
    '''
    # reading data
    if combpolboo:
        NRB_tra = nrbdic['NRB_tra']
        SNR_tra = nrbdic['SNR_tra']
    else:
        NRB_tra = nrbdic['NRB2_tra']  # co-pol
        SNR_tra = nrbdic['SNR2_tra']
    r_trm = nrbdic['r_trm']
    ts_ta = nrbdic['Timestamp']

    # retrieving scattering profile
    try:                        # scanning lidar NRB
        setz_a = nrbdic['DeltNbintheta_a']
        setzind_ta = nrbdic['DeltNbinthetaind_ta']
        z_tra = nrbdic['z_tra']
        theta_ta = nrbdic['theta_ta']
    except KeyError:            # vertical lidar NRB
        setz_a = nrbdic['DeltNbin_a']
        setzind_ta = nrbdic['DeltNbinind_ta']
        z_tra = nrbdic['r_tra']

    betaprimem_raa = np.array(list(map(
        lambda x: rayleigh_gen(*x), setz_a
    )))[:, -1, :]

    betamprime_tra = np.array(list(map(
        lambda x: betaprimem_raa[x], setzind_ta
    )))

    # computing gcdm mask
    gcdm_trm = np.arange(r_trm.shape[1])\
        < (np.argmax((SNR_tra <= NOISEALTITUDE) * r_trm, axis=1)[:, None])
    gcdm_trm *= r_trm

    # computing first derivative
    CRprime_tra = NRB_tra / betamprime_tra
    dzCRprime_tra = gradient_calc(CRprime_tra, z_tra, setz_a, setzind_ta)

    # finding clouds
    gcdm_algo(dzCRprime_tra, CRprime_tra, z_tra, gcdm_trm)


    # # plot feature
    # if plotboo:
    #     fig, (ax, ax1) = plt.subplots(ncols=2, sharey=True)
    #     for i, z_ra in enumerate(z_tra):
    #         if i == 100:
    #             print(np.rad2deg(theta_ta[i]))
    #             gcdm_rm = gcdm_trm[i]
    #             dzCRprime_plot = ax.plot(dzCRprime_tra[i][gcdm_rm], z_ra[gcdm_rm],
    #                                      marker='o')
    #             ax.vlines([amax_ta[i], amin_ta[i]], ymin=0, ymax=z_tra.max(),
    #                       color=dzCRprime_plot[0].get_color(), linestyle='--')
    #             ax1.plot(CRprime_tra[i], z_ra)
    #             ax1.vlines([bmax_ta[i]], ymin=0, ymax=z_tra.max(),
    #                        color=dzCRprime_plot[0].get_color(), linestyle='--')

    #     ax.set_ylim([0, 5])
    #     ax1.set_xscale('log')
    #     plt.show()


    # returning
    '''got to think about this'''



if __name__ == '__main__':
    from ...nrb_calc import nrb_calc
    from ....file_readwrite import smmpl_reader

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200901/202009010500.mpl',
        # starttime=LOCTIMEFN('202009010000', UTCINFO),
        # endtime=LOCTIMEFN('202009010800', UTCINFO),
        genboo=True,
    )

    main(nrb_d, combpolboo=True, plotboo=True)
