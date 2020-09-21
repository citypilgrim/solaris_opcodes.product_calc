# imports
import numpy as np

from ...constant_profiles import rayleigh_gen
from ....global_imports.solaris_opcodes import *

# params


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
    _, _, betamprime_tra, _, _, delfbetamprimes_tra = [
        tra[0]
        for tra in np.hsplit(rayleigh_tara, rayleigh_tara.shape[1])
    ]

    # computing CRprime
    CRprime_tra = NRB_tra / betamprime_tra
    delfCRprimes_tra = SNR_tra**-2 + delfbetamprimes_tra
    delCRprime_tra = CRprime_tra * np.sqrt(delfCRprimes_tra)

    # clear-sky search
    N_tra = (UCDMEPILSON**2) * delfCRprimes_tra



# testing
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
