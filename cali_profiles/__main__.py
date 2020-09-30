# imports
import os.path as osp

import pandas as pd
import numpy as np

from .deadtime_genread import main as deadtime_genread
from .afterpulse_csvgen import main as afterpulse_gen  # legacy code
from .overlap_csvgen import main as overlap_gen  # legacy code
# from .afterpulse_mplgen import main as afterpulse_gen
# from .overlap_mplgen import main as overlap_gen
from ...global_imports.solaris_opcodes import *


# params
_genfuncverb_boo = False


# main func
@verbose
@announcer
def main(
        lidarname,
        Delt, Nbin,
        mplreader=None,
        genboo=False, writeboo=False,
        deadtimedir=None, afterpulsedir=None, overlapdir=None,
        plotboo=False,
):
    '''
    Generates profiles for deadtime correction factor, overlap and afterpulse.
    It utilises the latest overlap and afterpulse file found in SOLARISMPLCALIDIR
    And speed of light constant found in solaris_opcodes.params
    range array calculated uses points in the middle of the range bin.
    range offset from range calibration applied only in solaris_opcodes.scan2ara

    This function generates calibration profiles interpolated to the specified
    bin sizes. Note any extrapolation is the same as the value at the extreme end
    It will read from file of the calculated profile if not genboo

    If genboo will utilise the latest afterpulse and overlap .mpl files, and
    deadtime.txt file to perform computations, performing interpolation of both
    data and uncertainty of data (afterpulse and overlap)

    Parameters
        lidarname (srt): directory name of lidar
        Delt (float): bintime
        Nbin (int): number of bins
        mplreader (func): either mpl_reader or smmpl_reader,
                          must be specified if genboo is True
        genboo (boolean): decides whether or not to go through the profile
                          generation check and potential calculation
        writeboo (boolean): determines whether to write the generated profiles
                            to a pre determined file
        deadtimedir (str): directory of .mpl file or generated calibration file
        afterpulsedir (str): directory of .mpl file or generated calibration file
        overlapdir (str): directory of .mpl file or generated calibration file
        plotboo (boolean): whether or not to show plots from afterpulse and
                           overlap calc, only works when genboo == True

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

        # finding latest file if not provided
        if not deadtimedir:
            D_dirlst = FINDFILESFN(DEADTIMEFILE,
                                   SOLARISMPLCALIDIR.format(lidarname))
            D_dirlst.sort(key=osp.getmtime)
            deadtimedir = D_dirlst[-1]
        if not afterpulsedir:
            napOE_dirlst = FINDFILESFN(AFTERPULSEFILE,
                                       SOLARISMPLCALIDIR.format(lidarname))
            napOE_dirlst.sort(key=DIRPARSEFN(AFTERPULSETIMEFIELD))
            afterpulsedir = napOE_dirlst[-1]
        if not overlapdir:
            Oc_dirlst = FINDFILESFN(OVERLAPFILE,
                                    SOLARISMPLCALIDIR.format(lidarname))
            Oc_dirlst.sort(key=DIRPARSEFN(OVERLAPTIMEFIELD))
            overlapdir = Oc_dirlst[-1]


        # getting file meta data
        Dsnstr = DIRPARSEFN(deadtimedir, DTSNFIELD)
        napOEdate = pd.Timestamp(DIRPARSEFN(afterpulsedir, AFTERPULSETIMEFIELD))
        Ocdate = pd.Timestamp(DIRPARSEFN(overlapdir, OVERLAPTIMEFIELD))

        # generatig profiles
        print('generating calibration files from:\n\t{}\n\t{}\n\t{}'.format(
            deadtimedir, afterpulsedir, overlapdir
        ))

        ## deadtime
        Dcoeff_a, D_func = deadtime_genread(deadtimedir, genboo=True,
                                            verbboo=_genfuncverb_boo)

        ## generating afterpulse and overlap correction profiles
        napOEr_ra, napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra =\
            afterpulse_gen(mplreader, afterpulsedir, D_func,
                           plotboo=plotboo, verbboo=_genfuncverb_boo)
        Ocr_ra, Oc_ra, delOc_ra =\
            overlap_gen(mplreader, overlapdir, D_func,
                        [napOEr_ra,
                         napOE1_ra, napOE2_ra,
                         delnapOE1_ra, delnapOE2_ra],
                        plotboo=plotboo, verbboo=_genfuncverb_boo)

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
        if writeboo:
            deadtimedir = DIRCONFN(CALIPROFILESDIR.format(lidarname),
                                   DEADTIMEPROFILE.format(Dsnstr))
            afterpulsedir = DIRCONFN(CALIPROFILESDIR.format(lidarname),
                                     AFTPROFILE.format(napOEdate, Delt, Nbin))
            overlapdir = DIRCONFN(CALIPROFILESDIR.format(lidarname),
                                  OVERPROFILE.format(Ocdate, Delt, Nbin))
            print('writing calibration files:\n\t{}\n\t{}\n\t{}'.format(
                deadtimedir, afterpulsedir, overlapdir
            ))
            np.savetxt(
                deadtimedir, Dcoeff_a,
                fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
            )
            np.savetxt(
                afterpulsedir, napOE_raa,
                fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
            )
            np.savetxt(
                overlapdir, Oc_raa,
                fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
            )

        # returning
        ret_l = [*napOE_raa, *Oc_raa, D_func]
        return ret_l

    # quick return of calibration output
    else:
        # retrieving latest file
        if not deadtimedir:
            D_dirlst = FINDFILESFN(
                DEADTIMEPROFILE, CALIPROFILESDIR.format(lidarname),
            )
            D_dirlst.sort(key=osp.getmtime)
            deadtimedir = D_dirlst[-1]
        if not afterpulsedir:
            afterpulsedir = FINDFILESFN(
                AFTPROFILE, CALIPROFILESDIR.format(lidarname),
                {AFTPROBINTIMEFIELD: Delt,
                 AFTPROBINNUMFIELD: Nbin}
            )[-1]
        if not overlapdir:
            overlapdir = FINDFILESFN(
                OVERPROFILE, CALIPROFILESDIR,
                {OVERPROBINTIMEFIELD: Delt,
                 OVERPROBINNUMFIELD: Nbin}
            )[-1]

        # read file
        print('reading calibration files from:\n\t{}\n\t{}\n\t{}'.format(
            deadtimedir, afterpulsedir, overlapdir
        ))
        _, D_func = deadtime_genread(deadtimedir, genboo=False)
        napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra =\
            np.loadtxt(afterpulsedir)
        Oc_ra, delOc_ra = np.loadtxt(overlapdir)

        # returning
        ret_l = [
            napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra,
            Oc_ra, delOc_ra,
            D_func
        ]
        return ret_l


# running
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ...file_readwrite import smmpl_reader

    # Delr ~15m, scan mini mpl
    napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra,\
        Oc_ra, delOc_ra,\
        D_func = main('smmpl_E2', 1e-7, 2000, smmpl_reader,
                      genboo=True, writeboo=False, plotboo=True)

    # testing
    Delr = SPEEDOFLIGHT * 1e-7
    r_ra = Delr * np.arange(2000) + Delr/2 + 0

    nap_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/measured_profiles/201910170400_2e-7_afterpulse.csv'
    Oc_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/measured_profiles/201910230900_2e-7_overlap.csv'
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
