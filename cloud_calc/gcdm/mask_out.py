# imports
import numpy as np
from scipy.interpolate import interp1d


# main func
def main(work_tra, z_ra, r_rm, setz):
    '''
    performs the mask r_rm on work_tra's range axis

    Parameters
        work_tra (np.ndarray): data array with 2 axis (time, range)
        z_ra (np.ndarray): range array to take derivative w.r.t
        r_rm (np.ndarray): corresponding mask array
        setz (tuple): descriptors for z_ra, contains Delt, Nbin, pad, theta
    Return
        retwork_tra (np.ndarray): mask out of work_tra
        retz_ra (np.ndarray): corresponding range array
        retr_rm (np.ndarray): corresponding mask array, all True
        setz (list): unchanged
    '''
    # applying mask
    retwork_tra = work_tra[:, r_rm]
    retz_ra = z_ra[r_rm]

    # changing setz
    setz[1] = retz_ra.shape[0]  # number of bins has changed

    return retwork_tra, retz_ra, np.ones_like(retz_ra, dtype=np.bool), setz


# testing
if __name__ == '__main__':
    main()
