# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from ...globalimports import *


# supp func
def _smooth_func(a, ra):
    # # convert negative values to zero
    # a[a<0] = 0
    # # apply filter
    # fa = savgol_filter(a, SAVGOLWINDOWLEN, SAVGOLPOLYORDER)
    # # trim
    # trimpos = np.argmax((np.abs(fa-a)<SAVGOLDIFFTHRES) * (ra>SAVGOLRANGETHRES))
    # if trimpos:
    #     a = a[:trimpos]
    #     fa = fa[trimpos:]
    #     return np.concatenate((a, fa))
    # else:
    #     return a
    return a



# main func
def main(
        mplreader, lidarname, mplfiledir, Dfunc,
        plotboo=False,
        slicetup=slice(AFTERPULSEPROFSTART, AFTERPULSEPROFEND, 1),
        compstr='a'
):
    '''
    This script looks for .mpl file (single file) related to afterpulse
    calibration measurements and churn out a profile based on the indicated bin
    time. Best practise is to utilise the same bin time as the afterpulse
    profile, so that we do not have to perform any interpolation

    Only the afterpulse profile is smoothed, the uncertainties of 'a' and 'b'
    are found to be smooth enough for interpolation.
    'b' uses the smooth profile for calc

    averaging time <= 1 min -> delE/E ~= DELEOVERE

    smoothening function is negated together with background count, left in the
    script incase it is useful in the future

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
        compstr (str): string describing how computation is carried out for
                       uncertainty.
                       'a': derived using my equations
                       'b': derived using campbell2002 uncertainty equations,
                            including first term
                       'c': derived using campbell2002 uncertainty equations,
                            excluding first term

    Return
        r_ra (np.array): range array, binsize given by .mpl file bin time
        napOE1/2_ra (np.array): normalised afterpulse correction
        delnapOE1/2s_ra (np.array): sqaure of uncert in normalised afterpulse
    '''
    # reading data
    mpl_dic = mplreader(
        lidarname,
        mplfiledir=mplfiledir, slicetup=slicetup
    )

    n1_tra = mpl_dic['Channel #1 Data']  # co-pol
    n2_tra = mpl_dic['Channel #2 Data']  # cross-pol
    n_trm = mpl_dic['Channel Data Mask']
    r_tra = mpl_dic['Range']

    E_ta = mpl_dic['Energy Monitor']
    N_ta = mpl_dic['Shots Sum']

    nb1_ta = mpl_dic['Background Average']
    nb2_ta = mpl_dic['Background Average 2']
    delnb1s_ta = mpl_dic['Background Std Dev']**2
    delnb2s_ta = mpl_dic['Background Std Dev 2']**2

    nb2_ta = np.zeros_like(nb2_ta)
    nb1_ta = np.zeros_like(nb1_ta)
    delnb2s_ta = np.zeros_like(delnb2s_ta)
    delnb1s_ta = np.zeros_like(delnb1s_ta)

    P1_tra = n1_tra * Dfunc(n1_tra)
    P2_tra = n2_tra * Dfunc(n2_tra)
    delP1s_tra = P1_tra / N_ta[:, None]
    delP2s_tra = P2_tra / N_ta[:, None]

    ## assuming afterpulse cali measurement consist of one ragne type
    r_ra = r_tra[0]
    n_rm = n_trm[0]

    # averaging in time
    if compstr == 'a':          # my method
        # calc afterpulse
        napOE1_tra = (P1_tra - nb1_ta[:, None])/E_ta[:, None]
        napOE2_tra = (P2_tra - nb2_ta[:, None])/E_ta[:, None]
        napOE1_ra = np.average(napOE1_tra, axis=0)
        napOE2_ra = np.average(napOE2_tra, axis=0)
        ## smoothening
        snapOE1_ra = _smooth_func(napOE1_ra, r_ra)
        snapOE2_ra = _smooth_func(napOE2_ra, r_ra)

        # calc uncert
        delnapOE1s_ra = 1/(napOE1_tra.shape[0]**2) * np.sum(
            (napOE1_tra**2) * (
                (delP1s_tra + delnb1s_ta[:, None])
                / ((P1_tra - nb1_ta[:, None])**2)
                + ((DELEOVERE**2) * np.ones(n1_tra.shape))
            ), axis=0
        )
        delnapOE2s_ra = 1/(napOE2_tra.shape[0]**2) * np.sum(
            (napOE2_tra**2) * (
                (delP2s_tra + delnb2s_ta[:, None])
                / ((P2_tra - nb2_ta[:, None])**2)
                + ((DELEOVERE**2) * np.ones(n1_tra.shape))
            ), axis=0
        )

    elif compstr == 'b':        # their method, including first term
        # calc afterpulse
        napOE1_ra = (
            np.average(P1_tra, axis=0) - np.average(nb1_ta)
        )/np.average(E_ta)
        napOE2_ra = (
            np.average(P2_tra, axis=0) - np.average(nb2_ta)
        )/np.average(E_ta)
        ## smoothening
        snapOE1_ra = _smooth_func(napOE1_ra, r_ra)
        snapOE2_ra = _smooth_func(napOE2_ra, r_ra)

        # calc uncert
        delnapOE1s_ra = (np.abs(snapOE1_ra)**2) * (
            1/n1_tra.shape[0] * (
                np.sqrt(np.sum(delP1s_tra, axis=0))
                + np.sqrt(np.sum(delnb1s_ta))
            )/((np.average(P1_tra, axis=0) - np.average(nb1_ta))**2)
            + (np.std(E_ta)/np.average(E_ta))**2
        )
        delnapOE2s_ra = np.abs(snapOE2_ra)**2 * (
            1/n2_tra.shape[0] * (
                np.sqrt(np.sum(delP2s_tra, axis=0))
                + np.sqrt(np.sum(delnb2s_ta))
            ) / ((np.average(P2_tra, axis=0) - np.average(nb2_ta))**2)
            + (np.std(E_ta)/np.average(E_ta))**2
        )
    elif compstr == 'c':        # their method
        # calc afterpulse
        napOE1_ra = (
            np.average(P1_tra, axis=0) - np.average(nb1_ta)
        )/np.average(E_ta)
        napOE2_ra = (
            np.average(P2_tra, axis=0) - np.average(nb2_ta)
        )/np.average(E_ta)
        ## smoothening
        snapOE1_ra = _smooth_func(napOE1_ra, r_ra)
        snapOE2_ra = _smooth_func(napOE2_ra, r_ra)

        # calc uncert
        delnapOE1s_ra = np.abs(snapOE1_ra) * (np.std(E_ta)/np.average(E_ta))**2
        delnapOE2s_ra = np.abs(snapOE2_ra) * (np.std(E_ta)/np.average(E_ta))**2

    else:
        raise ValueError('compstr = "a", "b" or "c"')

    if plotboo:
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
        # plotting raw profiles
        for i, n1_ra in enumerate(n1_tra):
            ax.plot(r_ra, n1_ra, color='C0')
            ax.plot(r_ra, n2_tra[i], color='C1')
        # plotting afterpulse
        ax1.plot(r_ra, napOE1_ra, 'x', color='C0')
        ax1.plot(r_ra, napOE2_ra, 'x', color='C1')
        ax1.plot(r_ra, snapOE1_ra, '-', color='b')
        ax1.plot(r_ra, snapOE2_ra, '-', color='orange')
        # plotting afterpulse uncert
        ax2.plot(r_ra, delnapOE1s_ra**0.5, '-', color='C0', label='cross')
        ax2.plot(r_ra, delnapOE2s_ra**0.5, '-', color='C1', label='co')
        # plot settings
        for axx in (ax, ax1, ax2):
            axx.set_yscale('log')
        ax.legend()
        ax.set_ylabel('raw counts')
        ax1.set_ylabel('afterpulse')
        ax2.set_ylabel('afterpulse uncert')
        plt.show()


    # applying mask and return
    ret_l = [r_ra, snapOE1_ra, snapOE2_ra, delnapOE1s_ra, delnapOE2s_ra]
    return list(map(lambda x:x[n_rm], ret_l))


# testing
if __name__ == '__main__':
    # imports
    import os.path as osp
    from glob import glob
    import pandas as pd
    from .deadtime_genread import main as deadtime_genread
    from ...file_readwrite import mpl_reader, smmpl_reader

    # writing to single file
    write2file_boo = False
    if write2file_boo:

        Delt = 2e-7
        Nbin = 2000

        smmpl_boo = False
        if smmpl_boo:
            lidarname, mpl_d = 'smmpl_E2', '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910170400_2e-7afterpulse.mpl'
            mplreader = smmpl_reader
        else:
            lidarname, mpl_d = 'mpl_S2S', '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/201907161201_5e-7afterpulse.mpl'
            mplreader = mpl_reader

        D_d = glob(
            CALIPROFILESDIR + '/*' + DEADTIMEPROFILE[DEADTIMEPROIND:]
            .format(lidarname)
        )[0]
        _, D_f = deadtime_genread(D_d, genboo=False)
        napOEr_ra, napOE1_ra, napOE2_ra, delnapOE1s_ra, delnapOE2s_ra =\
            main(mplreader, lidarname, mpl_d, D_f)
        # interpolating
        Delr = SPEEDOFLIGHT * Delt
        r_ra = Delr * np.arange(Nbin) + Delr/2
        napOE_raa = np.array([
            np.interp(r_ra, napOEr_ra, napOE1_ra),  # cross-pol
            np.interp(r_ra, napOEr_ra, napOE2_ra),  # co-pol
            np.interp(r_ra, napOEr_ra, delnapOE1s_ra**0.5),  # uncert cross-pol
            np.interp(r_ra, napOEr_ra, delnapOE2s_ra**0.5),  # uncert co-pol
        ])
        # writing to file
        napOEdate = pd.Timestamp(osp.basename(mpl_d)[:AFTERPULSETIMEIND])
        napOE_fn = AFTERPULSEPROFILE.format(napOEdate, Delt,
                                            Nbin, lidarname)
        np.savetxt(
            osp.join(CALIPROFILESDIR, napOE_fn),
            napOE_raa, fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
        )


    # checking which approach is more accurate
    compare_boo = True
    if compare_boo:

        lidarname, mpl_fn = 'mpl_S2S', '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/201909231105_5e-7afterpulse.mpl'

        D_d = glob(
            CALIPROFILESDIR + '/*' + DEADTIMEPROFILE[DEADTIMEPROIND:]
            .format(lidarname)
        )[0]
        _, D_f = deadtime_genread(D_d, genboo=False)

        # plotting to test
        fig, (ax3, ax4) = plt.subplots(nrows=2, sharex=True)

        ## sigmaMPL's values
        nap_fn = '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/201909231105_5e-7afterpulse.csv'
        r_ra, napOE2_ra, napOE1_ra = pd.read_csv(nap_fn, header=1).to_numpy().T
        ax3.plot(r_ra, napOE1_ra,  'kx', label='SigmaMPL')
        ax3.plot(r_ra, napOE2_ra, 'kx')


        ## calculated values
        for comp_str in ['a', 'b', 'c']:
            r_ra, napOE1_ra, napOE2_ra, delnapOE1s_ra, delnapOE2s_ra = main(
                mpl_reader, lidarname, mpl_fn, D_f,
                plotboo=False,
                compstr=comp_str
            )
            p = ax3.plot(r_ra, napOE2_ra, '-', label=f'{comp_str=}:napOE')
            ax3.plot(r_ra, napOE1_ra, '-', color=p[0].get_color())
            p = ax4.plot(r_ra, delnapOE2s_ra**0.5)
            ax4.plot(r_ra, delnapOE1s_ra**0.5, color=p[0].get_color())



        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax3.legend()
        plt.show()
