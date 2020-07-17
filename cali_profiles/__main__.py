# imports
from glob import glob
import os.path as osp

import pandas as pd
import numpy as np

from .deadtime_genread import main as deadtime_genread
from .afterpulse_csvgen import main as afterpulse_gen
from .overlap_csvgen import main as overlap_gen
# from .afterpulse_mplgen import main as afterpulse_gen # not implemented
# from .overlap_mplgen import main as overlap_gen
from ...globalimports import *


# supp func
def _tsfromfn_ff(timeind):
    def tsfromfn_f(fn):
        return pd.Timestamp(osp.basename(fn)[:timeind])
    return tsfromfn_f

# main func
@verbose
@announcer
def main(
        lidarname,
        Delt, Nbin,
        genboo=False, mplreader=None,
        plotboo=False, verbboo=True,
):
    '''
    Generates profiles for deadtime correction factor, overlap and afterpulse.
    It utilises the latest overlap and afterpulse file found in SOLARISMPLCALIDIR
    And speed of light constant found in solaris_opcodes.params
    range array calculated uses points in the middle of the range bin.
    range offset from range calibration applied only in solaris_opcodes.scan2ara

    This function generates calibration profiles interpolated to the specified
    bin sizes, moving the old files. Note any extrapolation is the same as the
    value at the extreme end
    It will read from file of the calculated profile if not genboo

    If genboo will utilise the latest afterpulse and overlap .mpl files, and
    deadtime.txt file to perform computations, performing interpolation of both
    data and uncertainty of data (afterpulse and overlap)


    Future
        - change generation of overlap and afterpulse files to be from .csv to
          .mpl under ...params, and also under the imports
        - move old calibration files to old_profiles

    Params
        lidarname (srt): directory name of lidar
        Delt (float): bintime
        Nbin (int): number of bins
        genboo (boolean): decides whether or not to go through the profile
                           generation check and potential calculation
        mplreader (func): either mpl_reader or smmpl_reader,
                          must be specified if genboo is True
        plotboo (boolean): whether or not to show plots from afterpulse and
                           overlap calc
        verboo (boolean): verbose

    Return
        napOE1_ra (array like): [MHz] afterpulse counts normalised by E
        delnapOE1_ra (array like): uncert of napOE_ra
        napOE2_ra (array like): [MHz] afterpulse counts normalised by E
        delnapOE2_ra (array like): uncert of napOE_ra
        Oc_ra (array like): overlap correction
        delOc_ra (array like): uncer of Oc_ra
        D_func (function): accepts counts array and output deadtime correction
    '''
    # generating profile
    if genboo:
        # retrieve filelist
        napOE_dirlst = glob(SOLARISMPLCALIDIR.format(lidarname)\
                          + '/*' + AFTERPULSEFILE[AFTERPULSENAMEIND:])
        Oc_dirlst = glob(SOLARISMPLCALIDIR.format(lidarname)\
                         + '/*' + OVERLAPFILE[OVERLAPNAMEIND:])
        D_dirlst = glob(SOLARISMPLCALIDIR.format(lidarname)\
                           + '/*' + DEADTIMEFILE[DEADTIMEMODELIND:])
        napOE_dirlst.sort(key=_tsfromfn_ff(AFTERPULSETIMEIND))
        Oc_dirlst.sort(key=_tsfromfn_ff(OVERLAPTIMEIND))
        D_dirlst.sort(key=osp.getmtime)
        ## retrieve latest file
        napOE_dir = napOE_dirlst[-1]
        napOEdate = pd.Timestamp(osp.basename(napOE_dir)[:AFTERPULSETIMEIND])
        Oc_dir = Oc_dirlst[-1]
        Ocdate = pd.Timestamp(osp.basename(Oc_dir)[:OVERLAPTIMEIND])
        D_dir = D_dirlst[-1]
        Dsnstr = osp.basename(D_dir)[:DEADTIMEMODELIND]

        # generatig profiles
        print('generating calibration files from:\n\t{}\n\t{}\n\t{}'.format(
            napOE_dir, Oc_dir, D_dir
        ))
        ## deadtime
        Dcoeff_a, D_func = deadtime_genread(D_dir, genboo=True)

        ## generating afterpulse and overlap correction profiles
        napOEr_ra, napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra =\
            afterpulse_gen(mplreader, lidarname, napOE_dir, D_func,
                           plotboo=plotboo)
        Ocr_ra, Oc_ra, delOc_ra = \
            overlap_gen(mplreader, lidarname, Oc_dir, D_func,
                        [napOEr_ra,
                         napOE1_ra, napOE2_ra,
                         delnapOE1_ra, delnapOE2_ra],
                        plotboo=plotboo)

        ## inter/extrapolate afterpulse and overlap
        Delr = SPEEDOFLIGHT * Delt
        r_ra = Delr * np.arange(Nbin) + Delr/2
        napOE_raa = np.array([
            np.interp(r_ra, napOEr_ra, napOE1_ra),  # cross-pol
            np.interp(r_ra, napOEr_ra, napOE2_ra),  # co-pol
            np.interp(r_ra, napOEr_ra, delnapOE1_ra),  # uncert cross-pol
            np.interp(r_ra, napOEr_ra, delnapOE2_ra),  # uncert co-pol
        ])
        Oc_raa = np.array([
            np.interp(r_ra, Ocr_ra, Oc_ra),
            np.interp(r_ra, Ocr_ra, delOc_ra),  # uncert
        ])

        # write to file
        Dcoeff_fn = DEADTIMEPROFILE.format(Dsnstr, lidarname)
        napOE_fn = AFTERPULSEPROFILE.format(napOEdate, Delt,
                                            Nbin, lidarname)
        Oc_fn = OVERLAPPROFILE.format(Ocdate, Delt, Nbin, lidarname)
        print('writing calibration files:\n\t{}\n\t{}\n\t{}'.format(
            Dcoeff_fn, napOE_fn, Oc_fn
        ))
        np.savetxt(
            osp.join(CALIPROFILESDIR, Dcoeff_fn),
            Dcoeff_a, fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
        )
        np.savetxt(
            osp.join(CALIPROFILESDIR, napOE_fn),
            napOE_raa, fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
        )
        np.savetxt(
            osp.join(CALIPROFILESDIR, Oc_fn),
            Oc_raa, fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
        )

        # returning
        return main(lidarname, Delt, Nbin, genboo=False, verbboo=True)

    # quick return of calibration output
    else:
        # retrieving filenames
        napOE_dir = glob(
            CALIPROFILESDIR + '/*' + AFTERPULSEPROFILE[AFTERPULSEPROTIMEIND:]
            .format(Delt, Nbin, lidarname)
        )[0]
        Oc_dir = glob(
            CALIPROFILESDIR + '/*' + OVERLAPPROFILE[OVERLAPPROTIMEIND:]
            .format(Delt, Nbin, lidarname)
        )[0]
        D_dir = glob(
            CALIPROFILESDIR + '/*' + DEADTIMEPROFILE[DEADTIMEPROIND:]
            .format(lidarname)
        )[0]

        # read file
        print('reading calibration files from:\n\t{}\n\t{}\n\t{}'.format(
            napOE_dir, Oc_dir, D_dir
        ))
        napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra = np.loadtxt(napOE_dir)
        Oc_ra, delOc_ra = np.loadtxt(Oc_dir)
        _, D_func = deadtime_genread(D_dir, genboo=False)

        ret_l = [
            napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra,
            Oc_ra, delOc_ra,
            D_func
        ]
        return ret_l

    
# running
if __name__ == '__main__':
    from ...file_readwrite import smmplfmt_dic
    ## Delr ~15m, scan mini mpl
    napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra,\
        Oc_ra, delOc_ra,\
        D_func = main('smmpl_E2', 1e-7/2, 2000,
                      genboo=True, plotboo=True)

    # testing
    Delr = SPEEDOFLIGHT * 1e-7/2
    r_ra = Delr * np.arange(2000) + Delr/2 + 0

    import matplotlib.pyplot as plt
    nap_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910170400_2e-7afterpulse.csv'
    Oc_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910230900_2e-7overlap.csv'
    onapOE_raa = pd.read_csv(nap_dir, header=1).to_numpy().T
    oOc_raa = pd.read_csv(Oc_dir, header=0).to_numpy().T

    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)
    ax.plot(onapOE_raa[0], onapOE_raa[1])  # scaling to match energy
    ax.plot(onapOE_raa[0], onapOE_raa[2])
    ax.plot(r_ra, napOE1_ra, 'o')
    ax.plot(r_ra, napOE2_ra, 'o')

    ax1.plot(oOc_raa[0], oOc_raa[1])
    ax1.plot(r_ra, Oc_ra, 'o')

    ax.set_yscale('log')
    plt.show()
