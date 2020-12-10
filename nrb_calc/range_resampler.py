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
    Does not change the padding of the arrays
    '''
    # mask
    mask = nrb_d[_maskkey]
    Nbin_a = (mask.sum(axis=-1)/rangestep).astype(np.int)
    pad_a = Nbin_a.max() - Nbin_a

    # DeltNbinpadtheta_a or DeltNbinpad_a
    try:
        Delt_a, _, _, theta_a = list(zip(*nrb_d['DeltNbinpadtheta_a']))
        Delt_a = np.array(Delt_a) * rangestep
        nrb_d['DeltNbinpadtheta_a'] = list(map(
            list, zip(Delt_a, Nbin_a, pad_a, theta_a)
        ))
        indkey = 'DeltNbinpadthetaind_ta'
    except KeyError:            # mpl files (no scanning) has no theta component
        Delt_a, _, _ = list(zip(*nrb_d['DeltNbinpad_a']))
        Delt_a = np.array(Delt_a) * rangestep
        nrb_d['DeltNbinpad_a'] = list(map(list, zip(Delt_a, Nbin_a, pad_a)))
        indkey = 'DeltNbinpadind_ta'

    # performing reshaping operation
    zsetind_ta = nrb_d[indkey]
    for key, func in _keypad_d.items():

        newa_a = []
        for i, a in enumerate(nrb_d[key]):
            m = mask[i]
            a = a[m]
            q, r = divmod(len(a), rangestep)
            if not r:
                sliceind = None
            else:
                sliceind = -r

            # reshaping
            newa = a[:sliceind]
            newa = newa.reshape(q, rangestep)
            newa = func(newa)

            # padding
            newa = np.append(
                np.zeros(pad_a[zsetind_ta[i]]),
                newa
            )

            newa_a.append(newa)

        nrb_d[key] = np.array(newa_a)

    # handling special cases

    ## mask
    nrb_d['r_trm'] = nrb_d['r_trm'].astype(np.bool)

    ## SNR
    nrb_d['SNR1_tra'] = nrb_d['NRB1_tra']/nrb_d['delNRB1_tra']
    nrb_d['SNR2_tra'] = nrb_d['NRB2_tra']/nrb_d['delNRB2_tra']
    nrb_d['SNR_tra'] = nrb_d['NRB_tra']/nrb_d['delNRB_tra']

    return nrb_d


# testing
if __name__ == '__main__':
    main()
