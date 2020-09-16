# imports
import numpy as np

from ....global_imports.solaris_opcodes import *


def _findbooisland_func(a, axis=-1):
    '''
    finds the start position of a boolean island in a 1d array
    '''
    absdiffa = np.abs(np.diff(a, axis=axis))
    inda = np.argwhere(absdiffa)




# main func
def main(dzCRprime_tra, CRprime_tra, z_tra, gcdm_trm):
    '''
    finds the cloud bottom and corresponding tops according to
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection

    Parameters
        CRprime_tra (np.ndarray): refer to paper, two dimensional first axis is
                                  time, second is altitude
        dzCRprime_tra (np.ndarray): first derivative of CRprime_tra
        z_tra (np.ndarray): altitude array, same dimensions as above
        gcdm_trm (np.ndarray): mask for array for gcdm algorithm

    Return
        gcdm_ta (np.ndarray): array of list of tuples. Each list signify a single
                              timestamp, each tuple within a list signifies a cloud
                              within each tuple is (cld bot [km], cld top [km])

                              If the corresponding cloud top for a given cloud
                              bottom is not detected, np.nan is placed.
    '''
    # Computing threshold
    CRprime0_tra = np.copy(CRprime_tra).flatten()
    CRprime0_tra[~(gcdm_trm.flatten())] = np.nan  # set to nan to ignore in average
    CRprime0_tra = CRprime0_tra.reshape(*(gcdm_trm.shape))
    barCRprime0_ta = np.nanmean(CRprime0_tra, axis=1)
    amax_ta = KEMPIRICAL * barCRprime0_ta
    amin_ta = (1 - KEMPIRICAL) * barCRprime0_ta

    # apply threshold
    ## finding cloud bases
    amaxcross_trm = (CRprime0_tra >= amax_ta[:, None])
    amincross_trm = (CRprime0_tra <= amin_ta[:, None])

    '''brute force compute the gcdm algorithm, but parrelisze the algorithm from
    each cloud bottom'''

# testing
if __name__ == '__main__':
    main()
