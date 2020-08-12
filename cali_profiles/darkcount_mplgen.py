# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from ...global_imports.solaris_opcodes import *


# main func
def main(
        mplreader, lidarname, mplfiledir, Dfunc,
        plotboo=False,
        slicetup=slice(DARKCOUNTPROFSTART, DARKCOUNTPROFEND, 1),
):
    '''
    This script looks for .mpl file (single file) related to darkcount
    calibration measurements and churn out a profile based on the indicated bin
    time. Best practise is to utilise the same bin time as the afterpulse
    profile, so that we do not have to perform any interpolation

    Assumes that background counts are close to zero

    averaging time <= 1 min -> delE/E ~= DELEOVERE

    Params
        mplreader (func): either mpl_reader or smmpl_reader
        lidarname (str): name of lidar
        mplfiledir (str): filename of mpl file to be read as afterpulse
                           calibration start of mpl file must be start of
                           measurement
        Dfunc (func): deadtime correction function
        plotboo (boolean): determines whether or not to plot the profiles chosen
                           for vetting
        slicetup (slice): slice tuple along time axis, only if mplfiledir
                          is specified

    Return
        r_ra (np.array): range array, binsize given by .mpl file bin time
        napOE1/2_ra (np.array): normalised afterpulse correction
        delnapOE1/2_ra (np.array): uncert in normalised afterpulse
    '''
    # reading data
    mpl_dic = mplreader(
        lidarname,
        mplfiledir=mplfiledir, slicetup=slicetup
    )

    n1_tra = mpl_dic['Channel #1 Data'] # co-pol
    n2_tra = mpl_dic['Channel #2 Data'] # cross-pol

    Delt_ta = mpl_dic['Bin Time'] # temporal size of bin
    Nbin_ta = mpl_dic['Number Data Bins']

    N_ta = mpl_dic['Shots Sum']
    ts_ta = mpl_dic['Timestamp']

    nb1_ta = mpl_dic['Background Average']
    delnb1_ta = mpl_dic['Background Std Dev']
    nb2_ta = mpl_dic['Background Average 2']
    delnb2_ta = mpl_dic['Background Std Dev 2']

    ## create range bins array; assuming all uniform
    Delr = SPEEDOFLIGHT * Delt_ta[0]  # binsize
    r_ra = Delr * np.arange(np.max(Nbin_ta)) + Delr/2

    P1_tra = n1_tra * Dfunc(n1_tra)
    delP1_tra = np.sqrt(P1_tra / N_ta[:, None])
    P2_tra = n2_tra * Dfunc(n2_tra)
    delP2_tra = np.sqrt(P2_tra / N_ta[:, None])

    p1_ta = np.average(P1_tra, axis=1)
    P2_ta = np.average(P2_tra, axis=1)
    timedim = len(N_ta)
    delP1_ta = np.sqrt(np.sum(delP1_tra**2, axis=1)) / timedim
    delP2_ta = np.sqrt(np.sum(delP2_tra**2, axis=1)) / timedim


    if plotboo:
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=False)
        # plotting raw profiles
        for i, n1_ra in enumerate(n1_tra):
            ax.plot(r_ra, n1_ra)
            ax.plot(r_ra, n2_tra[i])
        # plotting afterpulse
        ax1.plot_date(ts_ta, P1_ta)
        ax1.plot_date(ts_ta, P2_ta)
        # plot_dateting afterpulse uncert
        ax2.plot_date(ts_ta, delP1_ta, label='cross')
        ax2.plot_date(ts_ta, delP2_ta, label='co')
        # plot settings
        for axx in (ax, ax1, ax2):
            axx.set_yscale('log')
        ax.legend()
        ax.set_ylabel('raw counts')
        ax1.set_ylabel('afterpulse')
        ax2.set_ylabel('afterpulse uncert')
        plt.show()

    # return
    return ts_ta, P1_ta, P2_ta, delP1_ta, delP2_ta


# testing
if __name__ == '__main__':
    '''
    Notes on calculation found in logbook dated 20200518
    '''
    # imports
    from .deadtime_genread import main as deadtime_genread
    from ...file_readwrite import mpl_reader

    # testing
    '''compare day and night deadcount measurements'''
    lidarname, mpl_fn = 'mpl_S2S', '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/202003261205_5e-7darkcount.mpl'

    D_d = FINDFILESFN(DEADTIMEPROFILE, CALIPROFILESDIR,
                      {DTLIDARNAMEFIELD: lidarname})[0]
    _, D_f = deadtime_genread(D_d, genboo=False)

    # plotting to test
    fig, (ax3, ax4) = plt.subplots(nrows=2, sharex=True)

    ts_ta, P1_ta, P2_ta, delP1_ta, delP2_ta = main(
        mpl_reader, lidarname, mpl_fn, D_f,
        plotboo=False,
    )
    ax3.plot(P2_ta, '-', label='night co')
    ax4.plot(delP2_ta)

    ax3.plot(P1_ta, '-', label='night cross')
    ax4.plot(delP1_ta)

    mpl_fn = '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/202003190310_5e-7darkcount.mpl'
    ts_ta, P1_ta, P2_ta, delP1_ta, delP2_ta = main(
        mpl_reader, lidarname, mpl_fn, D_f,
        plotboo=False,
    )
    ax3.plot(P2_ta, '-', label='day co')
    ax4.plot(delP2_ta)

    ax3.plot(P1_ta, '-', label='day cross')
    ax4.plot(delP1_ta)

    ax3.legend()
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    plt.show()
