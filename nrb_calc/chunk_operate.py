# imports
import multiprocessing as mp

import numpy as np


# params


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
    process by first splitting the time axis into chunks that have similar z_ra.

    Parameters
        work_tra (np.ndarray): working array
        z_tra (np.ndarray): relevant range array
        r_trm (np.ndarray): mask array
        setz_a (list): list of parameter tuples containings the set of tuples to
                       fully describe the z_tra
        setzind_ta (np.ndarray): array of indexes for each array in z_tra to the
                                 setz_a
        func (function): operation to apply on range axis. Has to take in 3
                         positional arguments, work_Tra, z_ra, r_rm. and return
                         3 arrays, ret_Tra, retz_ra and retr_rm
        procnum (int): if specified, will perform the operations using
                       multiprocesses
        args (iterable): for func
        kwargs (dict): for func

    Return
        ret_tra (np.ndarray): array that has been operated on in the range axis
        retz_tra (np.ndarray): corresponding range array
        retr_trm (np.ndarray): corresponding mask
    '''
    alen = len(setz_a)

    # splitting into chunks
    pos_tra = np.arange(z_tra.shape[0])[:, None] * np.ones(z_tra.shape[1])
    setzind_tma = np.array([setzind_ta == i for i in range(alen)])
    _a3Tra = np.array([           # (alen, 3, chunk len(varies), Nbin-1)
        [                       # captial 'T' represent chunks of time or unsorted
            pos_tra[setzind_tm],
            work_tra[setzind_tm],
            z_tra[setzind_tm],
        ] for i, setzind_tm in enumerate(setzind_tma)
    ])
    _3aTra = np.transpose(_a3Tra, axes=[1, 0])  # (3, alen, chunk len(varies), Nbin)
    pos_aTra, work_aTra, z_aTra = _3aTra

    pos_ta = np.concatenate([   # array of positions for unsorted concat chunks
        pos_Tra[:, 0] for pos_Tra in pos_aTra
    ]).astype(np.int)
    pos_ta = np.argsort(pos_ta, kind='heapsort')  # array of indices for sorting

    z_ara = np.array([          # (alen, Nbin)
        z_tra[setzind_tm][0]
        for i, setzind_tm in enumerate(setzind_tma)
    ])
    r_arm = np.array([          # (alen, Nbin)
        r_trm[setzind_tm][0]
        for i, setzind_tm in enumerate(setzind_tma)
    ])

    # performing operation
    if procnum:                 # multi process
        pool = mp.Pool(processes=procnum)
        _a3a = [
            pool.apply(
                func,
                args=(work_Tra, z_ara[i], r_arm[i], *args),
                kwds=kwargs
            )
            for i, work_Tra in enumerate(work_aTra)
        ]
        pool.close()
        pool.join()
    else:                       # single process
        _a3a = [
            func(
                work_Tra, z_ara[i], r_arm[i],
                *args, **kwargs
            ) for i, work_Tra in enumerate(work_aTra)
        ]

    work_aTra = []
    z_ara = []
    r_arm = []
    for _3a in _a3a:
        work_aTra.append(_3a[0])
        z_ara.append(_3a[1])
        r_arm.append(_3a[2])

    # padding



    # concat and sorting







# testing
if __name__ == '__main__':
    main()
