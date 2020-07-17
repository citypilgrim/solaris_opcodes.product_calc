# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ...globalimports import *


# supp func
def _linear_f(x, m, c):
    return m * x + c

def _linearreg_ff(xa):
    def linearreg_f(ya):
        popt, pcov = curve_fit(_linear_f, xa, ya)
        return np.array([popt]), pcov

    return linearreg_f


# main func
def main(mplreader,
         lidarname,
         mplfiledir,
         Dfunc,
         napOEraa,
         combpolboo=True,
         plotboo=False,
         slicetup=slice(OVERLAPPROFSTART, OVERLAPPROFEND, 1),
         compstr='f'):
    '''
    This script looks for .mpl file (single file) related to overlap
    calibration measurements and churn out a profile based on the indicated bin
    time. Best practise is to utilise the same bin time as the overlap
    profile and afterpulse, so that we do not have to perform any interpolation

    averaging time <= 1 min -> delE/E ~= DELEOVERE

    Final profile is an average of each individual timestamp. Uncertainties are
    propagated by the average

    Future
        - remove testing and finish up return
        - Decide whether to calculate for co and both toggle
        - add plotting function
        - remove 2020 overlap calibration file from data dir to avoid using SG
          data

    Params
        mplreader (func): either mpl_reader or smmpl_reader
        lidarname (str): name of lidar
        mplfiledir (str): filename of mpl file to be read as afterpulse
                           calibration start of mpl file must be start of
                           measurement
        Dfunc (func): deadtime correction function
        napOEraa (np.array): non interpolated afterpulse array containing
                             range, normalised afterpulse and assoc. uncert
        plotboo (boolean): determines whether or not to plot the profiles chosen
                           for vetting
        slicetup (slice): slice tuple along time axis, only if mplfiledir
                          is specified
        compstr (str): string describing how computation is carried out for
                       uncertainty.
                       'a': curve fit on average, uncert from curve_fit func
                       'b': average of curve fits on each profile, error
                            propagation of uncert from curve fit
                       'c': curve fit on average, uncert from campbell2002
                       'd': average of curve fits on each profile, error
                            propagation of uncert from campbell2002
                       'e': curve fit on average, uncert from campbell2002
                            except del(ln Cbeta) term
                       'f': average of curve fits on each profile, error
                            propagation of uncert from campbell2002 except
                            del(ln Cbeta) term

    Return
        r_ra (np.array): range array, binsize given by .mpl file bin time
        Oc_ra (np.array): overlap correction
        delOc_ra (np.array): uncert in overlap
    '''
    # reading data
    mpl_dic = mplreader(lidarname, mplfiledir=mplfiledir, slicetup=slicetup)

    n1_tra = mpl_dic['Channel #1 Data']  # co-pol
    n2_tra = mpl_dic['Channel #2 Data']  # cross-pol

    E_ta = mpl_dic['Energy Monitor']
    N_ta = mpl_dic['Shots Sum']

    Delt_ta = mpl_dic['Bin Time']  # temporal size of bin
    Nbin_ta = mpl_dic['Number Data Bins']

    nb1_ta = mpl_dic['Background Average']
    nb2_ta = mpl_dic['Background Average 2']
    delnb1s_ta = mpl_dic['Background Std Dev']**2
    delnb2s_ta = mpl_dic['Background Std Dev 2']**2

    napOEr_ra, napOE1_ra, napOE2_ra, delnapOE1s_ra, delnapOE2s_ra = napOEraa

    ## create range bins array; assuming all uniform
    Delr = SPEEDOFLIGHT * Delt_ta[0]  # binsize
    r_ra = Delr * np.arange(np.max(Nbin_ta)) + Delr / 2

    ## other derived values
    napOE1_ra = np.interp(r_ra, napOEr_ra, napOE1_ra)
    napOE2_ra = np.interp(r_ra, napOEr_ra, napOE2_ra)

    P1_tra = n1_tra * Dfunc(n1_tra)
    P2_tra = n2_tra * Dfunc(n2_tra)
    delP1s_tra = P1_tra / N_ta[:, None]
    delP2s_tra = P2_tra / N_ta[:, None]


    # retreiving data based on specified polatisation
    if combpolboo:
        P_tra = P1_tra + P2_tra
        delPs_tra = delP1s_tra + delP2s_tra
        napOE_ra = napOE1_ra + napOE2_ra
        delnapOEs_ra = delnapOE1s_ra + delnapOE2s_ra
        nb_ta = nb1_ta + nb2_ta
        delnbs_ta = delnb1s_ta + delnb2s_ta
    else:
        P_tra = P2_tra
        delPs_tra = delP2s_tra
        napOE_ra = napOE1_ra
        delnapOEs_ra = delnapOE2s_ra
        nb_ta = nb2_ta
        delnbs_ta = delnb2s_ta


    # Computing Oc
    PH_tra = (r_ra[None, :]**2) * (
        (P_tra - nb_ta[:, None]) / E_ta[:, None]
        - napOE_ra[None, :]
    )
    lnPH_tra = np.log(PH_tra)


    ## slice params for linear regg
    r0boo_ra = r_ra >= OVERLAPSTARTTHRES
    sboo_ra = r0boo_ra * (r_ra <= OVERLAPENDTHRES)
    sr_ra = r_ra[sboo_ra]

    rlen = len(r_ra)
    r0pos = np.argmax(r0boo_ra)

    ## computing Oc_ra
    if compstr in ['b', 'd', 'f']:  # average on linear regression
        lrlnPH_tra = np.apply_along_axis(_linearreg_ff(sr_ra), 1,
                                         lnPH_tra[:, sboo_ra])
        lrlnPHpopt_ta = np.concatenate(lrlnPH_tra[:, 0])
        lrlnPHpcov_ta = lrlnPH_tra[:, 1]
        m2sigma_ta, lnCbeta_ta = lrlnPHpopt_ta[:, 0], lrlnPHpopt_ta[:, 1]
        PF_tra = np.exp(lnCbeta_ta[:, None] + m2sigma_ta[:, None] * r_ra)
        Oc_tra = PH_tra / PF_tra
        Oc_ra = np.average(Oc_tra, axis=0)

    elif compstr in ['a', 'c', 'e']:  # linear regression on average
        PH_ra, lnPH_ra = np.average([PH_tra, lnPH_tra], axis=1)
        lrlnPHpopt, lrlnPHpcov = curve_fit(
            _linear_f, r_ra[sboo_ra], lnPH_ra[sboo_ra]
        )
        m2sigma, lnCbeta = lrlnPHpopt
        PF_ra = np.exp(lnCbeta + m2sigma * r_ra)
        Oc_ra = PH_ra / PF_ra

    else:
        raise ValueError('compstr = "a", "b", "c", "d", "e" or "f"')

    ## concat constant 1 after r = r0
    Oc_ra = np.concatenate((Oc_ra[:r0pos], np.ones(rlen - r0pos)), axis=0)


    # computing del Oc
    ## computing delPH term
    delPHOPHs_tra = (
        (
            delPs_tra + delnbs_ta[:, None]
            + (napOE_ra * DELEOVERE * E_ta[:, None])**2
            + delnapOEs_ra * (E_ta[:, None])**2
        ) / ((P_tra - nb_ta[:, None] - napOE_ra * E_ta[:, None])**2)
        + DELEOVERE**2
    )
    if compstr in ['a', 'c', 'e']:  # error from each term
        delPHOPHs_ra = ((len(delPHOPHs_tra) * PH_ra)**-2) * np.sum(
            delPHOPHs_tra * (PH_tra**2), axis=0
        )

    ## computing differing term
    if compstr == 'a':
        delm2sigma_ta, dellnCbeta_ta = np.sqrt(np.diag(lrlnPHpcov_ta))
        '''compute error propagation for exponential'''

    elif compstr == 'b':
        delm2sigma, dellnCbeta = np.sqrt(np.diag(lrlnPHpcov))        

    elif compstr == 'c':
        pass

    elif compstr == 'd':
        pass

    elif compstr == 'e':
        pass

    elif compstr == 'f':
        pass

    # plotting linear regression for show
    if plotboo:
        # fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True)
        if compstr in ['b', 'd', 'f']:
            for i, lnPH_ra in enumerate(lnPH_tra):
                # p = ax.plot(r_ra, lnPH_ra, 'x')
                # ax.plot(r_ra, _linear_f(r_ra, m2sigma_ta[i], lnCbeta_ta[i]),
                #         color=p[0].get_color())
                plt.plot(r_ra, Oc_tra[i], 'x')
            plt.plot(r_ra, Oc_ra, 'ko')
        else:
            # p = ax.plot(r_ra, lnPH_ra, 'x')
            # ax.plot(r_ra, _linear_f(r_ra, m2sigma, lnCbeta),
            #         color=p[0].get_color())
            plt.plot(r_ra, Oc_ra, 'k-')

        # ax1.set_yscale('log')
        plt.yscale('log')
        # plt.show()

    # return
    return r_ra, Oc_ra#, delOc_ra


if __name__ == '__main__':
    '''
    Testing
        - Check that interpolation of uncertainty does not vary too largely
          from original uncertainty
    '''
    # imports
    from glob import glob
    from .deadtime_genread import main as deadtime_genread
    from .afterpulse_mplgen import main as afterpulse_mplgen
    from ...file_readwrite import smmpl_reader, mpl_reader

    # testing
    mplreader = smmpl_reader
    lidarname = 'smmpl_E2'
    D_d = glob(
        CALIPROFILESDIR + '/*' + DEADTIMEPROFILE[DEADTIMEPROIND:]
        .format(lidarname)
    )[0]
    _, D_f = deadtime_genread(D_d, genboo=False)

    mpl_d = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910170400_2e-7afterpulse.mpl'
    napOE_raa = afterpulse_mplgen(mplreader, lidarname, mpl_d, D_f)

    mpl_d = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910230900_2e-7overlap.mpl'
    main(mplreader, lidarname, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='b')
    main(mplreader, lidarname, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='a')

    '''plot SigmaMPL's plot'''

    plt.show()
