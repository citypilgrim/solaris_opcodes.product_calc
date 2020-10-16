# imports
import multiprocessing as mp

import numpy as np


# main func
def main(
        work_tra, z_tra, r_trm,
        setz_a, setzind_ta,
        func,
        *args,
        procnum=0,
        **kwargs,
):
    '''
    performs an operation on the range axis of the work_tra, but optimises the
    process by first splitting the time axis into chunks that have similar z_ra

    The mask array provided here MUST share the same mask for the same chunk

    Parameters
        work_tra (np.ndarray): working array
        z_tra (np.ndarray): relevant range array
        r_trm (np.ndarray): mask array
        setz_a (list): list of parameter lists containings the set of tuples to
                       fully describe the z_tra
        setzind_ta (np.ndarray): array of indexes for each array in z_tra to the
                                 setz_a
        func (function): operation to apply on range axis. Has to take in 3
                         positional arguments, work_Tra, z_ra, r_rm, setz.
                         must return the following:
                             1. ret_Tra (chunk length, range) working array
                             2. retz_ra (range), altitude array
                             3. retr_rm (range), mask array
                             4. retsetz, descriptor for altitude, padding unchanged
                         3 arrays, ret_Tra, retz_ra and retr_rm and 1 tuple of setz
        procnum (int): if specified, will perform the operations using
                       multiprocesses
        args (iterable): for func
        kwargs (dict): for func

    Return
        ret_tra (np.ndarray): array that has been operated on in the range axis
        retz_tra (np.ndarray): corresponding range array
        retr_trm (np.ndarray): corresponding mask
        retsetz_a (np.ndarray): new descriptor list for the different altitude arrays
                                in z_tra
    '''
    alen = len(setz_a)

    # splitting into chunks
    pos_tra = np.arange(z_tra.shape[0])[:, None] * np.ones(z_tra.shape[1])
    setzind_aTm = np.array([setzind_ta == i for i in range(alen)])
    _a3Tra = np.array([           # (alen, 3, chunk len(varies), Nbin-1)
        [                       # captial 'T' represent chunks of time or unsorted
            pos_tra[setzind_Tm],
            work_tra[setzind_Tm],
            z_tra[setzind_Tm],
        ] for i, setzind_Tm in enumerate(setzind_aTm)
    ])
    _3aTra = np.transpose(_a3Tra, axes=[1, 0])  # (3, alen, chunk len(varies), Nbin)
    pos_aTra, work_aTra, z_aTra = _3aTra

    pos_ta = np.concatenate([   # array of positions for unsorted concat chunks
        pos_Tra[:, 0] for pos_Tra in pos_aTra
    ]).astype(np.int)
    pos_ta = np.argsort(pos_ta, kind='heapsort')  # array of indices for sorting

    z_ara = np.array([          # (alen, Nbin)
        z_tra[setzind_Tm][0]
        for i, setzind_Tm in enumerate(setzind_aTm)
    ])
    r_arm = np.array([          # (alen, Nbin)
        r_trm[setzind_Tm][0]
        for i, setzind_Tm in enumerate(setzind_aTm)
    ])

    # performing operation
    if procnum:                 # multi process
        pool = mp.Pool(processes=procnum)
        _a3a = [
            pool.apply(
                func,
                args=(work_Tra, z_ara[i], r_arm[i], setz_a[i], *args),
                kwds=kwargs
            )
            for i, work_Tra in enumerate(work_aTra)
        ]
        pool.close()
        pool.join()
    else:                       # single process
        _a3a = [
            func(
                work_Tra, z_ara[i], r_arm[i], setz_a[i],
                *args, **kwargs
            ) for i, work_Tra in enumerate(work_aTra)
        ]

    ret_aTra = []
    retz_ara = []
    retr_arm = []
    retsetz_a = []
    for _3a in _a3a:
        ret_aTra.append(_3a[0])
        retz_ara.append(_3a[1])
        retr_arm.append(_3a[2])
        retsetz_a.append(_3a[3])

    # padding
    Tlen_a = []
    rlen_a = []

    ## getting padding dimensions
    for ret_Tra in ret_aTra:
        Tlen, rlen = ret_Tra.shape
        Tlen_a.append(Tlen)
        rlen_a.append(rlen)
    padlen = max(rlen_a)
    rlen_a = [padlen - rlen for rlen in rlen_a]

    ### adjusting setz for new padding
    retsetz_a = []
    for i, setz in enumerate(setz_a):
        retsetz = setz
        retsetz[2] = rlen_a[i]
        retsetz_a.append(retsetz)

    ## pad
    for i, ret_Tra in enumerate(ret_aTra):
        Tlen, rlen = Tlen_a[i], rlen_a[i]
        ret_aTra[i] = np.concatenate((np.zeros((Tlen, rlen)), ret_Tra), axis=1)
        retz_ara[i] = np.concatenate((np.zeros(rlen), retz_ara[i]))
        retr_arm[i] = np.concatenate((np.zeros(rlen, dtype=np.bool), retr_arm[i]))

    ## concat
    ret_tra = np.concatenate(ret_aTra)  # only this needs to be sorted
    retz_tra = np.array([retz_ara[setzind] for setzind in setzind_ta])
    retr_trm = np.array([retr_arm[setzind] for setzind in setzind_ta])

    # sorting
    if alen != 1:               # skips sorting if only one type of range array
        ret_tra = ret_tra[pos_ta]

    return ret_tra, retz_tra, retr_trm, retsetz_a



# testing
if __name__ == '__main__':
    main()
