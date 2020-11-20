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

    work_tra = NRB_tra/betamprime_tra  # CRprime

    # getting products from extended GCDM
    gcdm_ta = gcdm_extended(
        r_trm, z_tra, setz_a, setzind_ta,
        work_tra,
        plotboo=False,
    )


    # returning
    return gcdm_ta



if __name__ == '__main__':
    '''refer to the individual gcdm protocols for plotting'''
    pass
