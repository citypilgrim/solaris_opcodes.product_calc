# imports
import numpy as np

from ....global_imports.solaris_opcodes import *


# main func
def main(
        dzCRprime_ra, gcdm_rm,
        amin, amax,
):
    '''
    finds the cloud bottom and corresponding tops according to
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection

    utilizes multiprocessing to apply the gcdm algorithm on the time axis

    Parameters
        dzCRprime_ra (np.ndarray): first derivative of CRprime, refer to paper
        gcdm_rm (np.ndarray): mask for GCDM
        amin/max (float): threshold values for GCDM

    Return
        gcdm_a (list): list of tuples. Each tuple within a list signifies a
                       cloud. Within each tuple is (cld bot ind, cld top ind)
                       indices are taken w.r.t to dzCRprime_ta after applying
                       the gcdm mask
                       If the corresponding cloud top for a given cloud
                       bottom is not detected, np.nan is placed.
    '''
    dzCRprime_ra = dzCRprime_ra[gcdm_rm]

    # finding cloud bases
    amaxcross_rm = (dzCRprime_ra >= amax)
    start_boo = amaxcross_rm[0]
    absdiff_a = np.abs(np.diff(amaxcross_rm, axis=-1))
    cloudbotind_a = np.argwhere(absdiff_a)[:, 0] + 1
    if start_boo:
        cloudbotind_a = np.insert(cloudbotind_a[1::2], 0, 0)
    else:
        cloudbotind_a = cloudbotind_a[::2]

    # finding cloud top
    cloudtopind_a = []
    for cloudbotind in cloudbotind_a:

        crossaminboo_a = dzCRprime_ra[cloudbotind:] < amin
        if crossaminboo_a.any():  # array val has decreased
            crossaminind = np.argmax(crossaminboo_a) + cloudbotind

            crossaminboo_a = dzCRprime_ra[crossaminind:] > amin
            if crossaminboo_a.any():
                cloudtopind = np.argmax(crossaminboo_a) + crossaminind
            else:
                cloudtopind = np.nan

        else:                   # array val did not decrease significantly
            cloudtopind = np.nan

        cloudtopind_a.append(cloudtopind)

    return list(zip(cloudbotind_a, cloudtopind_a))


# testing
if __name__ == '__main__':
    main()
