# imports
import numpy as np


# params
_maskkey = 'r_trm'

_sum_func = lambda x: x.sum(axis=-1)
_avg_func = lambda x: x.mean(axis=-1)
_mult_func = lambda x: x.prod(axis=-1)
_takefirst_func = lambda x: x[..., 0]
_adduncert_func = lambda x: np.sqrt((x**2).sum(axis=-1))/x.shape[-1]

_keypad_d = {                   # operations on array which have been
    'NRB1_tra': _sum_func,      # reshaped into (time, range resampled, rangestep)
    'NRB2_tra': _sum_func,
    'NRB_tra': _sum_func,

    'delNRB1_tra': _adduncert_func,
    'delNRB2_tra': _adduncert_func,
    'delNRB_tra': _adduncert_func,

    'r_tra': _takefirst_func,
    'z_tra': _takefirst_func,

    'r_trm': _mult_func
}


# main func
def main(nrb_d, rangestep):
    '''
    Due to possible unequal lengths of the arrays, the arrays are treated
    individually and padded from the rear with zeros.
    This also applies to the net mask

    Only pads the required amount
    '''
    # performing reshaping operation
    mask = list(nrb_d[_maskkey])
    for key, func in _keypad_d.items():

        alen_a = np.array([])
        newa_a = []
        for i, a in enumerate(nrb_d[key]):
            a = a[mask[i]]
            q, r = divmod(len(a), rangestep)
            if not r:
                sliceind = None
            else:
                sliceind = -r
            newa = a[:sliceind]
            newa = newa.reshape(q, rangestep)
            newa = func(newa)

            newa_a.append(newa)
            alen_a = np.append(alen_a, q)

        # padding
        padlen_a = (np.max(alen_a) - alen_a).astype(np.int)
        newa_a = np.array([
            np.append(newa, np.zeros(padlen_a[j]))
            for j, newa in enumerate(newa_a)
        ])

        nrb_d[key] = np.array(newa_a)

    # handling special cases

    ## mask
    nrb_d['r_trm'] = nrb_d['r_trm'].astype(np.bool)

    ## SNR
    nrb_d['SNR1_tra'] = nrb_d['NRB1_tra']/nrb_d['delNRB1_tra']
    nrb_d['SNR2_tra'] = nrb_d['NRB2_tra']/nrb_d['delNRB2_tra']
    nrb_d['SNR_tra'] = nrb_d['NRB_tra']/nrb_d['delNRB_tra']

    ## DeltNbintheta_a or DeltNbin_a
    Delt_a, Nbin_a = list(zip(*nrb_d['DeltNbin_a']))
    Delt_a = np.array(Delt_a) * rangestep
    Nbin_a = alen_a
    nrb_d['DeltNbin_a'] = list(zip(Delt_a, Nbin_a))

    try:
        Delt_a, Nbin_a, theta_a = list(zip(*nrb_d['DeltNbintheta_a']))
        Delt_a = np.array(Delt_a) * rangestep
        Nbin_a = alen_a
        nrb_d['DeltNbintheta_a'] = list(zip(Delt_a, Nbin_a, theta_a))
    except KeyError:            # mpl files (no scanning) has no theta component
        pass

    return nrb_d


# testing
if __name__ == '__main__':
    main()
