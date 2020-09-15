# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

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
    r_trm = nrbdic['r_trm']
    ts_ta = nrbdic['Timestamp']

    # retrieving scattering profile
    try:                        # scanning lidar NRB
        setz_a = nrbdic['DeltNbintheta_a']
        alen = len(setz_a)
        setzind_ta = nrbdic['DeltNbinthetaind_ta']
        z_tra = nrbdic['z_tra']
    except KeyError:            # vertical lidar NRB
        setz_a = nrbdic['DeltNbin_a']
        alen = len(setz_a)
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
    zspace_taa = np.diff(z_tra, axis=1)
    dzCRprime_tra = np.diff(CRprime_tra, axis=1) / zspace_taa
    dzz_tra = zspace_taa/2 + z_tra[:, :-1]

    # interpolating derivative to maintain same z_tra
    ## splitting into chunks based on Delt, Nbin, theta
    pos_tra = np.arange(dzz_tra.shape[0])[:, None] * np.ones(dzz_tra.shape[1])
    Traa = np.array([           # (alen, 3, chunk len, Nbin-1)
        [                       # captial 'T' represent chunks of time or unsorted
            pos_tra[setzind_ta == i],
            dzCRprime_tra[setzind_ta == i],
            dzz_tra[setzind_ta == i],
        ] for i in range(alen)
    ])           # dtype is object, containing the last two dimensions
    Traa = np.transpose(Traa, axes=[1, 0])  # (3, alen, chunk len, Nbin)
    pos_Traa, dzCRprime_Traa, dzz_Traa = Traa
    for pos_Tra in pos_Traa:
        print(pos_Tra.shape)
    '''PROBLEM Traa is an array containing arrays of different chunk length, i.e. it cannot be made into an array'''

    pos_Ta = np.concatenate(pos_Traa[..., 0]).astype(np.int)
    z_raa = np.array([
            z_tra[setzind_ta == i][0] for i in range(alen)
    ])
    ## interpolating chunks; same dzr within chunk
    dzCRprime_Traa = [
        interp1d(               # returns a function
            dzz_Tra[0], dzCRprime_Traa[i], axis=1,
            kind='quadratic', fill_value='extrapolate'
        )(z_raa[i]) for i, dzz_Tra in enumerate(dzz_Traa)
    ]
    ## joining chunks together and sorting
    dzCRprime_Tra = np.concatenate(dzCRprime_Traa)
    if alen != 1:               # skip sorting if they are all have same range
        dzCRprime_tra = dzCRprime_Tra[pos_Ta]
    else:
        dzCRprime_tra = dzCRprime_Tra

    # Computing threshold
    dzCRprime0_tra = np.copy(dzCRprime_tra).flatten()
    dzCRprime0_tra[~(gcdm_trm.flatten())] = 0
    dzCRprime0_tra = dzCRprime0_tra.reshape(*(gcdm_trm.shape))
    barCRprime0_ta = np.average(dzCRprime0_tra, axis=1)
    amax_ta = KEMPIRICAL * barCRprime0_ta
    amin_ta = (1 - KEMPIRICAL) * barCRprime0_ta


    # apply threshold
    ## finding cloud bases
    amaxcross_trm = (dzCRprime0_tra >= amax_ta[:, None])
    amincross_trm = (dzCRprime0_tra <= amin_ta[:, None])
    '''check with measurement data if the double peak really does occur'''


    # plot feature
    if plotboo:
        fig, ax = plt.subplots()
        for i, r_ra in enumerate(z_tra):
            gcdm_rm = gcdm_trm[i]
            dzCRprime0_plot = ax.plot(
                dzCRprime0_tra[i][gcdm_rm], r_ra[gcdm_rm])
            ax.vlines([amax_ta[i], amin_ta[i]], ymin=0, ymax=z_tra.max(),
                      color=dzCRprime0_plot[0].get_color())
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
