# imports
import numpy as np
from scipy.integrate import simps


# main func
def main(
        owork_ra, work_ra,
        z_ra, gcdm_rm,
        gcdm_a,
        integral_threshold
):
    '''
    performs a cloud layer check on gcdm_a using an integral comparison between
    the unfiltered data and filtered data

    Parameters
        owork_ra (np.ndarray): unfiltered working array
        work_ra (np.ndarray): filtered working array
        z_ra (np.ndarray): altitude array
        gcdm_rm (np.ndarray): corresponding mask
        integral_threshold (float): thresholding value for integral comparison
    Return
        gcdm_a (np.ndarray): (no. of layers, 2(cldbot, cldtop)) after removing
                             invalid layers
    '''
    # applying mask
    owork_ra = owork_ra[gcdm_rm]
    work_ra = work_ra[gcdm_rm]
    z_ra = z_ra[gcdm_rm]

    # iterating through cloud layers
    rm_l = []
    for i, gcdm_2a in enumerate(gcdm_a):

        # finding integral range
        cldbot, cldtop = gcdm_2a
        cldbotind = np.argmax(z_ra >= cldbot)
        if np.isnan(cldtop):
            cldtopind = None
        else:
            cldtopind = np.argmax(z_ra >= cldtop)

        iowork_ra = owork_ra[cldbotind:cldtopind]
        iwork_ra = work_ra[cldbotind:cldtopind]
        iz_ra = z_ra[cldbotind:cldtopind]

        iowork_ra[iowork_ra<0] = 0
        # iwork_ra[iwork_ra<0] = 0

        # integrating
        oworkint = simps(iowork_ra, iz_ra)
        workint = simps(iwork_ra, iz_ra)

        # deciding whether to remove anot
        if abs((workint - oworkint)/workint) > integral_threshold:
            rm_l.append(i)

    # removing
    gcdm_a = np.delete(gcdm_a, rm_l, axis=0)

    return gcdm_a


# testing
if __name__ == '__main__':
    main()
