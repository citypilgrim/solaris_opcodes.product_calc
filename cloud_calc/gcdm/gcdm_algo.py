# imports
import numpy as np

from ....global_imports.solaris_opcodes import *


# main func
def main(
        CRprime_ra, dzCRprime_ra,
        z_ra, gcdm_rm,
        amin, amax,
):
    '''
    finds the cloud bottom and corresponding tops according to
    Gradient-based Cloud Detection (GCDM) according to Lewis et. al 2016.
    Overview of MPLNET Version 3 Cloud Detection

    utilizes multiprocessing to apply the gcdm algorithm on the time axis

    Parameters
        CRprime_ra (np.ndarray): NRB normalised by molecular profile
        dzCRprime_ra (np.ndarray): first derivative of CRprime, refer to paper
        z_ra (np.ndarray): corresponding altitude array
        gcdm_rm (np.ndarray): mask for GCDM
        amin/max (float or np.ndarray): threshold values for GCDM along range axis

    Return
        gcdm_a (np.ndarray): nested array. Each inner array signifies a cloud.
                             inner array = [cld bot, cld top]
                             indices are taken w.r.t to dzCRprime_ta after applying
                             the gcdm mask
                             If the corresponding cloud top for a given cloud
                             bottom is not detected, np.nan is placed.
    '''
    CRprime_ra = CRprime_ra[gcdm_rm]
    dzCRprime_ra = dzCRprime_ra[gcdm_rm]
    z_ra = z_ra[gcdm_rm]
    lenz = z_ra.shape[0]
    try:
        amin = amin[gcdm_rm]
        amax = amax[gcdm_rm]
    except IndexError:
        pass

    # finding cloud bases
    amaxcross_rm = (dzCRprime_ra >= amax)
    try:
        start_boo = amaxcross_rm[0]
    except IndexError:          # no clouds found
        return []

    absdiff_a = np.abs(np.diff(amaxcross_rm, axis=-1))
    cloudbotind_a = np.argwhere(absdiff_a)[:, 0] + 1
    if start_boo:
        cloudbotind_a = np.insert(cloudbotind_a[1::2], 0, 0)
    else:
        cloudbotind_a = cloudbotind_a[::2]
    cloudbotz_a = z_ra[cloudbotind_a]

    # finding cloud top
    cloudtopind_a = np.array([])
    cloudtopz_a = np.array([])
    for cloudbotind in cloudbotind_a:

        crossaminboo_a = (dzCRprime_ra < amin)[cloudbotind:]
        if crossaminboo_a.any():  # array val has decreased
            crossaminind = np.argmax(crossaminboo_a) + cloudbotind

            crossaminboo_a = (dzCRprime_ra > amin)[crossaminind:]
            if crossaminboo_a.any():
                cloudtopind = np.argmax(crossaminboo_a) + crossaminind
                cloudtopz = z_ra[cloudtopind]
            else:               # array val did not cross above amin
                cloudtopind = lenz
                cloudtopz = np.nan

        else:                   # array val did not decrease enough, cld top not found
            cloudtopz = np.nan
            cloudtopind = lenz

        cloudtopz_a = np.append(cloudtopz_a, cloudtopz)
        cloudtopind_a = np.append(cloudtopind_a, cloudtopind)

    # finding cloud peak

    ## creating rectangular array
    cloudslice_zla = (np.arange(lenz)[:, None] >= cloudbotind_a) \
        * (np.arange(lenz)[:, None] < cloudtopind_a)
    cloudslice_zla = CRprime_ra[:, None] * cloudslice_zla
    cloudpeakind_a = np.argmax(cloudslice_zla, axis=0)
    cloudpeakz_a = z_ra[cloudpeakind_a]

    gcdm_a = np.stack([cloudbotz_a, cloudpeakz_a, cloudtopz_a], axis=1)
    return gcdm_a


# testing
if __name__ == '__main__':
    main()
