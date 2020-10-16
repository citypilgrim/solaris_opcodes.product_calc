# imports
import numpy as np
from scipy.interpolate import interp1d


# main func
def main(work_tra, z_ra, r_rm, setz):
    '''
    Computes the first derivative of work_tra w.r.t z_ra

    Parameters
        work_tra (np.ndarray): data array with 2 axis (time, range)
        z_ra (np.ndarray): range array to take derivative w.r.t
        r_rm (np.ndarray): corresponding mask array
        setz (tuple): descriptors for z_ra, contains Delt, Nbin, pad, theta
    Return
        dzwork_tra (np.ndarray): first derivative of work_tra
        retz_ra (np.ndarray): corresponding range array
        retr_rm (np.ndarray): corresponding mask array, all True
        setz (list): unchanged
    '''
    # applying mask
    dzwork_tra = work_tra[:, r_rm]
    retz_ra = z_ra[r_rm]

    # first derivative
    dzwork_tra = np.diff(dzwork_tra, axis=1) / (retz_ra[1] - retz_ra[0])
    dzretz_ra = np.diff(retz_ra)/2 + retz_ra[:-1]

    # interpolating derivative to maintain same z_ra
    dzwork_tra = interp1d(
        dzretz_ra, dzwork_tra, axis=1,
        kind='quadratic', fill_value='extrapolate'
    )(retz_ra)

    return dzwork_tra, retz_ra, np.ones_like(retz_ra, dtype=np.bool), setz


# testing
if __name__ == '__main__':
    main()
