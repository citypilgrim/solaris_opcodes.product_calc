# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *


# supp func
def _aaaraygen_func(dim1arr, **kwargs):
    '''
    Parameters
        kwargs (dict)
    '''
    return rayleigh_gen(*dim1arr, **kwargs)


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
        - correct the error where we are not passing altitude to the rayleigh_gen
          profile

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
    r_tra = nrbdic['r_tra']
    r_trm = nrbdic['r_trm']
    ts_ta = nrbdic['Timestamp']

    # retrieving scattering profile
    try:                        # scanning lidar NRB
        setz_a = nrbdic['DeltNbintheta_a']
        alen = len(setz_a)
        setzind_ta = nrbdic['DeltNbinthetaind_ta']
    except KeyError:            # vertical lidar NRB
        setz_a = nrbdic['DeltNbin_a']
        alen = len(setz_a)
        setzind_ta = nrbdic['DeltNbinind_ta']

    print(np.array(setz_a).shape)
    '''NOTSURE WHY APPLY_ALONG_AXIS IS NOT WORKING'''
    betaprimem_raa = np.apply_along_axis(
        _aaaraygen_func, 1, np.array(setz_a),
    )
    print(betaprimem_raa.shape)
    betamprime_tra = np.array(list(map(lambda x: betaprimem_raa[x],
                                       setzind_ta)))


    # computing gcdm mask
    gcdm_trm = np.arange(r_trm.shape[1])\
        < (np.argmax((SNR_tra <= NOISEALTITUDE) * r_trm, axis=1)[:, None])
    gcdm_trm *= r_trm

    # computing first derivative
    Cbetaprime_tra = NRB_tra / betamprime_tra
    rspace_taa = np.diff(r_tra, axis=1)
    drCbetaprime_tra = np.diff(Cbetaprime_tra, axis=1) / rspace_taa
    drr_tra = rspace_taa/2 + r_tra[:, :-1]


    for ind in range(0, len(drr_tra), 10):
        fig, (ax, ax1) = plt.subplots(ncols=2, sharey=True)
        print(f'we are at {ind}, time is {ts_ta[ind]}')
        ax1.plot(Cbetaprime_tra[ind][gcdm_trm[ind]],
                 r_tra[ind][gcdm_trm[ind]])
        ax.plot(drCbetaprime_tra[ind][gcdm_trm[ind][:-1]],
                drr_tra[ind][gcdm_trm[ind][:-1]])
        drCbetaprime0_tra = np.copy(drCbetaprime_tra).flatten()
        drCbetaprime0_tra[~(gcdm_trm[:, :-1].flatten())] = 0
        drCbetaprime0_tra = drCbetaprime0_tra.reshape(*(gcdm_trm[:, :-1].shape))
        barCbetaprime0_ta = np.average(drCbetaprime0_tra, axis=1)
        amax_ta = KEMPIRICAL * barCbetaprime0_ta
        amin_ta = (1 - KEMPIRICAL) * barCbetaprime0_ta
        ax.vlines([amax_ta[ind], amin_ta[ind]], ymin=0, ymax=r_tra.max())

        plt.show()
    '''check NRB profiles on why NAN for the following indexes'''
    '''
    90
    '''

    # interpolating derivative to maintain same r_tra
    ## splitting into chunks based on Delt and Nbin
    pos_tra = np.arange(len(drr_tra))[:, None] * np.ones(len(drr_tra[0]))
    Traa = np.array([           # (alen, 3, Delt snippets, Nbin)
        [                       # captial 'T' chunks of time or unsorted
            pos_tra[DeltNbinind_ta == i],
            drCbetaprime_tra[DeltNbinind_ta == i],
            drr_tra[DeltNbinind_ta == i],
        ] for i in range(alen)
    ])
    Traa = np.transpose(Traa, axes=[1, 0, 2, 3])# (3, alen, Delt snippets, Nbin)
    pos_Traa, drCbetaprime_Traa, drr_Traa = Traa
    pos_Ta = np.concatenate(pos_Traa[..., 0]).astype(np.int)
    r_raa = np.array([
            r_tra[DeltNbinind_ta == i][0] for i in range(alen)
    ])
    ## interpolating chunks; same drr within chunk
    drCbetaprime_Traa = [
        interp1d(               # returns a function
            drr_Tra[0], drCbetaprime_Traa[i], axis=1,
            kind='quadratic', fill_value='extrapolate'
        )(r_raa[i]) for i, drr_Tra in enumerate(drr_Traa)
    ]
    ## joining chunks together and sorting
    drCbetaprime_Tra = np.concatenate(drCbetaprime_Traa)
    if alen != 1:               # skip sorting if they are all have same range
        drCbetaprime_tra = drCbetaprime_Tra[pos_Ta]
    else:
        drCbetaprime_tra = drCbetaprime_Tra

    # Computing threshold
    drCbetaprime0_tra = np.copy(drCbetaprime_tra).flatten()
    drCbetaprime0_tra[~(gcdm_trm.flatten())] = 0
    drCbetaprime0_tra = drCbetaprime0_tra.reshape(*(gcdm_trm.shape))
    barCbetaprime0_ta = np.average(drCbetaprime0_tra, axis=1)
    amax_ta = KEMPIRICAL * barCbetaprime0_ta
    amin_ta = (1 - KEMPIRICAL) * barCbetaprime0_ta


    # apply threshold
    ## finding cloud bases
    amaxcross_trm = (drCbetaprime0_tra >= amax_ta[:, None])
    amincross_trm = (drCbetaprime0_tra <= amin_ta[:, None])
    '''check with measurement data if the double peak really does occur'''


    # plot feature
    if plotboo:
        fig, ax = plt.subplots()
        for i, r_ra in enumerate(r_tra):
            gcdm_rm = gcdm_trm[i]
            drCbetaprime0_plot = ax.plot(
                drCbetaprime0_tra[i][gcdm_rm], r_ra[gcdm_rm])
            ax.vlines([amax_ta[i], amin_ta[i]], ymin=0, ymax=r_tra.max(),
                      color=drCbetaprime0_plot[0].get_color())
        # ax.set_ylim([0, 3])
        ax.set_xlim([-10, 10])
        plt.show()


    # returning
    '''got to think about this'''



if __name__ == '__main__':
    from ...nrb_calc import nrb_calc
    from ....file_readwrite import smmpl_reader

    nrb_d = nrb_calc(
        'smmpl_E2', smmpl_reader,
        '/home/tianli/SOLAR_EMA_project/codes/solaris_opcodes/product_calc/nrb_calc/testNRB_smmpl_E2.mpl',
        genboo=True,
    )

    main(nrb_d, combpolboo=True, plotboo=True)
