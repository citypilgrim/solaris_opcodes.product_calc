# imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ...global_imports.solaris_opcodes import *


# supp func
def _linear_f(x, m, c):
    return m * x + c

def _linearreg_ff(xa):
    def linearreg_f(ya):
        popt, pcov = curve_fit(_linear_f, xa, ya)
        return np.array([popt]), pcov

    return linearreg_f


fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True)


# main func
def main(mplreader,
         mplfiledir,
         Dfunc,
         napOEraa,
         combpolboo=True,
         plotboo=False,
         slicetup=slice(OVERLAPPROFSTART, OVERLAPPROFEND, 1),
         compstr='f'):
    '''
    churn out overlap profile based on the indicated bin time.
    Best practise is to utilise the same bin time as the overlap
    profile and afterpulse, so that we do not have to perform any interpolation

    Equations here are based on campbell 2002 Micopulse Lidar Signals: Uncertainty
    Analysis

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
                       'c': average of curve fits on each profile, error
                            propagation of uncert from campbell2002
                       'd': average of curve fits on each profile, error
                            propagation of uncert from campbell2002 except
                            del(ln Cbeta) term

    Return
        r_ra (np.array): range array, binsize given by .mpl file bin time
        Oc_ra (np.array): overlap correction
        delOc_ra (np.array): uncert in overlap
    '''
    # reading data
    mpl_dic = mplreader(mplfiledir=mplfiledir, slicetup=slicetup)

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
    delP1s_tra = P1_tra / N_ta[:, None]  # 's' at the end means squared
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

    ## performing linear regression and computing value

    ### average on linear regression
    if compstr in ['b', 'c', 'd']:
        lrlnPH_tra = np.apply_along_axis(_linearreg_ff(sr_ra), 1,
                                         lnPH_tra[:, sboo_ra])
        lrlnPHpopt_ta = np.concatenate(lrlnPH_tra[:, 0])
        lrlnPHpcov_ta = np.stack(lrlnPH_tra[:, 1], axis=0)
        m2sigma_ta, lnCbeta_ta = lrlnPHpopt_ta[:, 0], lrlnPHpopt_ta[:, 1]
        lnPF_tra = lnCbeta_ta[:, None] + m2sigma_ta[:, None] * r_ra
        PF_tra = np.exp(lnPF_tra)
        Oc_tra = PH_tra / PF_tra
        Oc_ra = np.average(Oc_tra, axis=0)

    ### linear regression on average;same as linear regression on all points
    elif compstr == 'a':
        PH_ra, lnPH_ra = np.average([PH_tra, lnPH_tra], axis=1)
        lrlnPHpopt, lrlnPHpcov = curve_fit(
            _linear_f, r_ra[sboo_ra], lnPH_ra[sboo_ra]
        )
        m2sigma, lnCbeta = lrlnPHpopt
        lnPF_ra = lnCbeta + m2sigma * r_ra
        PF_ra = np.exp(lnPF_ra)
        Oc_ra = PH_ra / PF_ra

    else:
        raise ValueError('compstr = "a", "b", "c", "d"')

    ## concat constant 1 after r = r0
    Oc_ra = np.concatenate((Oc_ra[:r0pos], np.ones(rlen - r0pos)), axis=0)


    # computing del Oc
    ## computing delPH term; 'SNR' means del<val>/<val>
    SNRPHs_tra = (
        (
            delPs_tra + delnbs_ta[:, None]
            + (napOE_ra * DELEOVERE * E_ta[:, None])**2
            + delnapOEs_ra * (E_ta[:, None])**2
        ) / ((P_tra - nb_ta[:, None] - napOE_ra * E_ta[:, None])**2)
        + DELEOVERE**2
    )

    ## computing differing term
    if compstr == 'a':          # uncert from linear regression
        delm2sigma, dellnCbeta = np.sqrt(np.diag(lrlnPHpcov))
        SNRem2sigmas_ra = (r_ra * delm2sigma) ** 2
        SNRCbetas = dellnCbeta ** 2

        SNRPHs_ra = ((len(SNRPHs_tra) * PH_ra)**-2) * np.sum(
            SNRPHs_tra * (PH_tra**2), axis=0
        )
        SNRPFs_ra = SNRem2sigmas_ra + SNRCbetas
        delOc_ra = Oc_ra * np.sqrt(SNRPHs_ra + SNRPFs_ra)

    elif compstr == 'b':        # error propagate uncert from linear regression
        delm2sigma_ta, dellnCbeta_ta = np.sqrt(np.array([
            np.diag(ara) for ara in lrlnPHpcov_ta
        ])).T

        SNRem2sigmas_tra = (r_ra * delm2sigma_ta[:, None]) ** 2
        SNRCbetas_ta = dellnCbeta_ta ** 2

        SNRPFs_tra = SNRem2sigmas_tra + SNRCbetas_ta[:, None]
        delOc_ra = (1/len(SNRPHs_tra)) * np.sqrt(np.sum(
            Oc_tra * (SNRPHs_tra + SNRPFs_tra), axis=0
        ))

    elif compstr == 'c':  # error propagate uncert from campbell2002 equations
        X = r_ra.size
        Omega = X * np.sum(r_ra**2) - np.sum(r_ra)**2
        ss_ta = 1/(X-2) * np.sum(np.nan_to_num(lnPH_tra - lnPF_tra)**2, axis=1)

        delm2sigma_ta = np.sqrt(X * ss_ta / Omega)
        SNRem2sigmas_tra = (r_ra * delm2sigma_ta[:, None]) ** 2

        dellnCbeta_ta = np.sqrt(ss_ta/Omega * np.sum(r_ra**2))
        SNRCbetas_ta = (
            (np.exp(lnCbeta_ta + dellnCbeta_ta) - np.exp(lnCbeta_ta))\
            + (np.exp(lnCbeta_ta) - np.exp(lnCbeta_ta - dellnCbeta_ta))
        ) / (2 * np.exp(lnCbeta_ta))

        SNRPFs_tra = SNRem2sigmas_tra + SNRCbetas_ta[:, None]
        delOc_ra = (1/len(SNRPHs_tra)) * np.sqrt(np.sum(
            Oc_tra * (SNRPHs_tra + SNRPFs_tra), axis=0
        ))

    elif compstr == 'd':
        X = r_ra.size
        Omega = X * np.sum(r_ra**2) - np.sum(r_ra)**2
        ss_ta = 1/(X-2) * np.sum(np.nan_to_num(lnPH_tra - lnPF_tra)**2, axis=1)

        delm2sigma_ta = np.sqrt(X * ss_ta / Omega)
        SNRem2sigmas_tra = (r_ra * delm2sigma_ta[:, None]) ** 2

        dellnCbeta_ta = np.sqrt(ss_ta/Omega * np.sum(r_ra**2))
        SNRCbetas_ta = dellnCbeta_ta ** 2

        SNRPFs_tra = SNRem2sigmas_tra + SNRCbetas_ta[:, None]
        delOc_ra = (1/len(SNRPHs_tra)) * np.sqrt(np.sum(
            Oc_tra * (SNRPHs_tra + SNRPFs_tra), axis=0
        ))

    ## trimming delOc after lower limit of regression
    delOc_ra[r0boo_ra] = 0


    # plotting linear regression for show
    if plotboo:
        if compstr in ['b', 'c', 'd']:
            for i, lnPH_ra in enumerate(lnPH_tra):
                p = ax.plot(r_ra, lnPH_ra, 'x')
                ax.plot(r_ra, _linear_f(r_ra, m2sigma_ta[i], lnCbeta_ta[i]),
                        color=p[0].get_color())
                ax1.plot(r_ra, Oc_tra[i], 'x', color=p[0].get_color())
            p = ax1.plot(r_ra, Oc_ra, linestyle='-', label=compstr)
            ax1.plot(r_ra, delOc_ra, linestyle='--', color=p[0].get_color())
        else:
            p = ax.plot(r_ra, lnPH_ra, 'kx')
            ax.plot(r_ra, _linear_f(r_ra, m2sigma, lnCbeta),
                    color=p[0].get_color())
            p = ax1.plot(r_ra, Oc_ra, linestyle='-', label=compstr)
            ax1.plot(r_ra, delOc_ra, linestyle='--', color=p[0].get_color())

        ax1.set_yscale('log')
        # plt.show()

    # return
    return r_ra, Oc_ra#, delOc_ra


# testing
if __name__ == '__main__':
    # imports
    from glob import glob
    from .deadtime_genread import main as deadtime_genread
    from .afterpulse_mplgen import main as afterpulse_mplgen
    from .afterpulse_csvgen import main as afterpulse_csvgen
    from .overlap_csvgen import main as overlap_csvgen
    from ...file_readwrite import smmpl_reader, mpl_reader

    # testing
    mplreader = smmpl_reader
    lidarname = 'smmpl_E2'

    ## computing deadtime
    D_d = FINDFILESFN(DEADTIMEFILE, SOLARISMPLCALIDIR.format(lidarname))[0]
    _, D_f = deadtime_genread(D_d, genboo=True)

    ## computing afterpulse
    mpl_d = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/measured_profiles/201910170400_2e-7_afterpulse.mpl'
    napOE_raa = afterpulse_mplgen(mplreader, mpl_d, D_f, compstr='c')

    ## computing overlap
    mpl_d = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/measured_profiles/201910230900_2e-7_overlap.mpl'
    main(mplreader, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='b')
    main(mplreader, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='c')
    main(mplreader, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='d')
    main(mplreader, mpl_d, D_f, napOE_raa,
         combpolboo=True, plotboo=True, compstr='a')


    ## plotting SigmaMPL's version
    csv_d = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/generated_profiles/201910230900_2e-7_overlap.csv'
    r_ra, Oc_ra, delOc_ra = overlap_csvgen(
        mplreader, csv_d, D_f, napOE_raa
    )
    # plt.plot(r_ra, Oc_ra, 'k-')


    plt.xlim([-1, 10])
    plt.legend()
    plt.show()
