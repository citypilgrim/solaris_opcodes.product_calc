# imports
import numpy as np

from ...global_imports.solaris_opcodes import *


# params
_average_l = [
    'NRB1_tra', 'NRB2_tra', 'NRB_tra',
]
_averageerrorprop_l = [
    'delNRB1_tra', 'delNRB2_tra', 'delNRB_tra',
]
_firstval_l = [
    'Timestamp',
    'r_tra',
    'z_tra', 'theta_ta', 'phi_ta',
    'DeltNbinpadind_ta',
    'DeltNbinpadthetaind_ta'
]
_unchanged_l = [
    'DeltNbinpad_a',
    'DeltNbinpadtheta_a',
    'theta_a'
]
_multiplication_l = [
    'r_trm'
]


# main func
@verbose
@announcer
def main(nrb_d, step):
    '''
    Returns the time averaged version of the NRB dict, averaging by step number of
    profiles in chronological order
    '''

    ret_d = {}

    for key in _unchanged_l:
        try:
            ret_d[key] = nrb_d[key]
        except KeyError:
            pass

    # trimming off remainders and reshaping into steps
    alen = len(nrb_d['Timestamp'])
    q, r = divmod(alen, step)
    if r:
        sliceind = -r
    else:
        sliceind = None
    for key in _average_l + _averageerrorprop_l + _firstval_l + _multiplication_l:
        try:
            ret_d[key] = nrb_d[key][:sliceind].reshape(
                (q, step, *(nrb_d[key].shape[1:]))
            )
        except KeyError:
            pass

    # handling each averaging case

    ## average
    for key in _average_l:
        try:
            ret_d[key] = ret_d[key].mean(axis=1)
        except KeyError:
            pass

    ## average error propagation
    for key in _averageerrorprop_l:
        try:
            ret_d[key] = np.sqrt((ret_d[key]**2).sum(axis=1)) / step
        except KeyError:
            pass

    ## first values
    for key in _firstval_l:
        try:
            ret_d[key] = ret_d[key][:, 0, ...]
        except KeyError:
            pass

    ## multiplying values
    for key in _multiplication_l:
        try:
            ret_d[key] = ret_d[key].prod(axis=1)
        except KeyError:
            pass


    # handling special cases

    ## SNR
    ret_d['SNR1_tra'] = ret_d['NRB1_tra'] / ret_d['delNRB1_tra']
    ret_d['SNR2_tra'] = ret_d['NRB2_tra'] / ret_d['delNRB2_tra']
    ret_d['SNR_tra'] = ret_d['NRB_tra'] / ret_d['delNRB_tra']

    ## boolean mask
    ret_d['r_trm'] = ret_d['r_trm'].astype(np.bool)


    return ret_d




# testing
if __name__ == '__main__':
    main()
