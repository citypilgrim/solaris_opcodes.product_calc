# imports
import numpy as np
from scipy.interpolate import interp1d


# main func
def main(CRprime_tra, z_tra, setz_a, setzind_ta):
    '''
    Computers the first derivative of CRprime_tra w.r.t z_tra, where z_tra is two
    dimensional, having one axis in time and the other in altitude.
    z_tra consists of ranges characterised by 3 parameters (Delt, Nbin, theta)

    Parameters
        CRprime_tra (np.ndarray): data array
        z_tra (np.ndarray): range array to take derivative w.r.t to,
        setz_a (list): list of parameter tuples containings the set of tuples to
                       fully describe the z_tra
        setzind_ta (np.ndarray): array of indexes for each array in z_tra to the
                                 setz_a
    Return
        dzCRprime_tra (np.ndarray): first derivative of CRprime_tra
    '''

    alen = len(setz_a)

    zspace_taa = np.diff(z_tra, axis=1)
    dzCRprime_tra = np.diff(CRprime_tra, axis=1) / zspace_taa
    dzz_tra = zspace_taa/2 + z_tra[:, :-1]

    # interpolating derivative to maintain same z_tra
    ## splitting into chunks based on Delt, Nbin, theta
    pos_tra = np.arange(dzz_tra.shape[0])[:, None] * np.ones(dzz_tra.shape[1])
    setzind_tma = np.array([setzind_ta == i for i in range(alen)])
    Traa = np.array([           # (alen, 3, chunk len(varies), Nbin-1)
        [                       # captial 'T' represent chunks of time or unsorted
            pos_tra[setzind_tm],
            dzCRprime_tra[setzind_tm],
            dzz_tra[setzind_tm],
        ] for i, setzind_tm in enumerate(setzind_tma)
    ])           # dtype is object, containing the last two dimensions
    Traa = np.transpose(Traa, axes=[1, 0])  # (3, alen, chunk len(varies), Nbin)
    pos_Traa, dzCRprime_Traa, dzz_Traa = Traa

    pos_Ta = np.concatenate([   # array of positions for unsorted concat chunks
        pos_Tra[:, 0] for pos_Tra in pos_Traa
    ]).astype(np.int)
    pos_Ta = np.argsort(pos_Ta, kind='heapsort')  # array of indices for sorting

    z_raa = np.array([          # (alen, Nbin)
        z_tra[setzind_tm][0]
        for i, setzind_tm in enumerate(setzind_tma)
    ])
    ## interpolating chunks; same dzz within chunk
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

    return dzCRprime_tra


# testing
if __name__ == '__main__':
    main()
