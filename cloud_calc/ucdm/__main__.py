# imports
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

# from .clearskysearch_conservative_algo import main as clearskysearch_algo
from .clearskysearch_liberal_algo import main as clearskysearch_algo
from .objthres_level1_func import main as objthres_func
# from .objthres_level2_func import main as objthres_func
from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *


# params
_cleaskysearch_pool = mp.Pool(processes=UCDMPROCNUM)


# main func
@verbose
@announcer
def main(
        nrbdic,
        combpolboo=True,
        plotboo=False,
):
    '''
    Uncertainty-based Cloud Detection (UCDM) according to Campbell et. al 2007.
    Elevated Cloud and Erosol Layer Retrievals from Micropulse Lidar Signal
    Profiles

    Here CRprime and Cstar from the paper are equivalent. In our computations
    we do not perform time averaging of our profiles

    Future
        - rayleigh profile computed does not have padding

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
    delNRB_tra = NRB_tra/SNR_tra
    r_trm = nrbdic['r_trm']

    # retrieving scattering profile
    try:                        # scanning lidar NRB
        setz_a = nrbdic['DeltNbinpadtheta_a']
        setzind_ta = nrbdic['DeltNbinpadthetaind_ta']
        z_tra = nrbdic['z_tra']
    except KeyError:            # static lidar NRB
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
    _, _, betamprime_tra, delfbetamprimes_tra = [
        tra[:, 0, :]
        for tra in np.hsplit(rayleigh_tara, rayleigh_tara.shape[1])
    ]

    # CRprime
    CRprime_tra = NRB_tra / betamprime_tra
    delfCRprimes_tra = SNR_tra**-2 + delfbetamprimes_tra
    delCRprime_tra = CRprime_tra * np.sqrt(delfCRprimes_tra)

    # no. significant bins
    N_tra = np.ceil((UCDMEPILSON**2) * delfCRprimes_tra).astype(np.int)

    # setting limit on computation range
    ucdm_trm = r_trm * (z_tra >= UCDMCLEARSKYALTITUDE)

    # clear-sky search
    clearskysearch_taa = np.array([
        _cleaskysearch_pool.apply(
            clearskysearch_algo,
            args=(CRprime_tra[i][ucdm_rm], delCRprime_tra[i][ucdm_rm],
                  N_tra[i][ucdm_rm], z_tra[i][ucdm_rm])
        )
        for i, ucdm_rm in enumerate(ucdm_trm)
    ])
    _cleaskysearch_pool.close()
    _cleaskysearch_pool.join()
    Cfstar_ta = clearskysearch_taa[:, 0]
    delCfstar_ta = clearskysearch_taa[:, -1]
    clearskybound_tba = clearskysearch_taa[:, -2:]

    # PAB
    PAB_tra = NRB_tra / Cfstar_ta[:, None]
    delPAB_tra = PAB_tra * np.sqrt(
        SNR_tra**-2 + (delCfstar_ta/Cfstar_ta)[:, None]**2
    )

    # objective threshold
    alpha_tra = objthres_func(
        delCfstar_ta, Cfstar_ta,
        betamprime_tra,
        delNRB_tra,
    )

    # particulate base layer mask
    baselayerheight_trm = (PAB_tra - delPAB_tra > alpha_tra) * ucdm_trm


    # plotting
    if plotboo:
        fig, (ax, ax1, ax2, ax3) = plt.subplots(ncols=4, sharey=True)
        ts_ta = nrbdic['Timestamp']
        theta_ta = nrbdic['theta_ta']

        for i, z_ra in enumerate(z_tra):
            # if i in range(a:=0, a+10):
            if i == 25:
                r_rm = r_trm[i]
                ucdm_rm = ucdm_trm[i]
                baselayerheight_rm = baselayerheight_trm[i]

                # checking some vital info
                print(ts_ta[i])
                delCRprime_val = delCRprime_tra[i][ucdm_rm]
                print(f'delCRprime stats: {delCRprime_val.mean()} +/- {delCRprime_val.std()}')
                print(f'max alt:{z_tra[i].max()}')
                print(f'theta:{np.rad2deg(theta_ta[i])}')
                print(f'max range:{z_tra[i].max()/np.cos(theta_ta[i])}')

                print(f'binsize:{(z_ra[1]-z_ra[0])/np.cos(theta_ta[i])}')

                # plot cloud search
                baselayerheight_ra = z_ra[baselayerheight_rm]
                p = ax3.plot(
                    np.zeros_like(baselayerheight_ra), baselayerheight_ra,
                    marker='o', linestyle=''
                )

                # plot thresholding
                # p = ax2.errorbar(
                #     PAB_tra[i][ucdm_rm], z_ra[ucdm_rm],
                #     xerr=delPAB_tra[i][ucdm_rm],
                #     fmt='o', linestyle='-',
                #     label='PAB'
                # )
                ax2.plot(
                    alpha_tra[i][ucdm_rm], z_ra[ucdm_rm],
                    color=p[0].get_color(), linestyle='', marker='o',
                    label='alpha'
                )
                ax2.plot(
                    betamprime_tra[i][ucdm_rm], z_ra[ucdm_rm],
                    color=p[0].get_color(), linestyle='dotted',
                    label='rayleigh'
                )

                # plot for clear sky search
                if clearskybound_tba.any():
                    print(clearskybound_tba[i])
                    for clearskybound in clearskybound_tba[i]:
                        ax1.axhline(
                            clearskybound,
                            color=p[0].get_color(), linestyle='--'
                        )
                p = ax1.errorbar(
                    CRprime_tra[i][ucdm_rm], z_ra[ucdm_rm],
                    xerr=delCRprime_tra[i][ucdm_rm],
                    fmt='o', linestyle=''
                )

                # plot unprocessed data
                ax.plot(NRB_tra[i][r_rm], z_ra[r_rm])
                # ax.plot(SNR_tra[i][r_rm], z_ra[r_rm])
                # ax.plot(N_tra[i][ucdm_rm], z_ra[ucdm_rm])


        ax.set_xscale('log')
        # ax1.set_xscale('log')

        ylowerlim, yupperlim = 0, 20
        ax.set_ylim([ylowerlim, yupperlim])
        # ax1.set_ylim([ylowerlim, yupperlim])
        xlowerlim, xupperlim = -8e3, 8e3
        # ax1.set_xlim([xlowerlim, xupperlim])

        ax3.legend()
        ax2.legend()
        ax1.legend()
        ax.legend()

        plt.show()


# testing
if __name__ == '__main__':
    from ...nrb_calc import main as nrb_calc
    from ....file_readwrite import smmpl_reader, mpl_reader

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        # '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200307/202003070300.mpl',
        # '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200901/202009010500.mpl',
        # '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200930/202009300400.mpl',
        '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200805/202008050003.mpl',
        # starttime=LOCTIMEFN('202009010000', 0),
        # endtime=LOCTIMEFN('202009010800', 0),
        timestep=None, rangestep=5,
    )

    # nrb_d = nrb_calc(
    #     'mpl_S2S', mpl_reader,
    #     '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/20200602/202006020000.mpl',
    #     timestep=5, rangestep=None,
    # )

    main(nrb_d, combpolboo=True, plotboo=True)
