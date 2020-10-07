# imports
import json
import numpy as np

from .range_resampler import main as range_resampler
from .time_average import main as time_average
from ..cali_profiles import cali_profiles
from ...file_readwrite import mpl_reader, smmpl_reader
from ...global_imports.solaris_opcodes import *


# params
_static_azimuth = 0             # [deg]
_static_elevation = 90          # [deg]


# supp func
def _aaacaliprofiles_func(dim1arr, args, kwargs):
    '''
    Parameters
        args (tuple): (lidarname,)
        kwargs (dict)
    '''
    return cali_profiles(*args, *dim1arr, **kwargs)

# main func
@verbose
@announcer
def main(
        lidarname, mplreader,
        mplfiledir=None,
        starttime=None, endtime=None,
        timestep=None, rangestep=None,
        genboo=True,
        writeboo=False,
):
    '''
    uncert in E ~= 1% assuming that measurement averages are typically <= 1min,
    which is equivalent to temperature fluctuations of <= 2% according to
    campbell 20002 uncertainty paper

    Parameters
        lidarname (str): directory name of lidar
        mplreader (func): either mpl_reader or smmpl_reader,
                          must be specified if genboo is True
        mplfiledir (str): mplfile to be processed if specified, date should be
                          None
        start/endtime (datetime like): approx start/end time of data of interest
        timestep (int): if specified, will return a time averaged version of the
                        original,
                        i.e. new timedelta = timedelta * timestep
        rangestep (int): if specified, will return a spatially resampled version
                         of the original,
                         i.e. new range bin size = range bin * rangestep
        genboo (boolean): if True, will read .mpl files and generate NRB, return
                          and write
                          if False, will read exisitng nrb files and only return
        writeboo (boolean): data is written to filename if True, ignored if genboo
                            is False.
                            Does not write time averaged data

    Return
        ret_d (dict):
            DeltNbinpad_a (list): set of zipped Delt_ta and Nbin_ta and pad_ta
            DeltNbinpadind_ta (np.array): DeltNbinpad_a indexes for the tra arrays
            r_tra (np.array): shape (time dim, no. range bins)
            r_trm (np.array): shape (time dim, no. range bins), usually all ones
            NRB/1/2_tra (np.array): shape (time dim, no. range bins)
            delNRB/1/2_tra (np.array): shape (time dim, no. range bins)
            SNR/1/2_tra (np.array): shape (time dim, no. range bins)

        if 'Azimiuth Angle' and 'Elevation Angle' are valid keys
            theta/phi_ta (np.array): [rad] if it exsists in the mplfile
                                     spherical coordinates of data, angular offset
                                     corrected
            theta_a (np.array): set of theta values
            z_tra (np.array): [km] altitude, if theta_ta exists
            DeltNbinpadtheta_a (list): set of zipped Delta_ta, Nbin_ta, pad_ta,
                                       theta_ta
            DeltNbinpadthetaind_ta (np.array): DeltNbinpad_a thetaindexes for the tra
                                            arrays
    '''
    # computation
    if genboo:
        # read .mpl files
        mpl_d = mplreader(
            DIRCONFN(SOLARISMPLDIR.format(lidarname)),
            mplfiledir=mplfiledir,
            starttime=starttime, endtime=endtime,
            filename=None,
        )

        ts_ta = mpl_d['Timestamp']

        n1_tra = mpl_d['Channel #1 Data']  # co-pol
        n2_tra = mpl_d['Channel #2 Data']  # cross-pol
        n_trm = mpl_d['Channel Data Mask']
        r_tra = mpl_d['Range']

        E_ta = mpl_d['Energy Monitor']
        N_ta = mpl_d['Shots Sum']

        Delt_ta = mpl_d['Bin Time']  # temporal size of bin
        Nbin_ta = mpl_d['Number Data Bins']
        pad_ta = mpl_d['Pad']   # front padding in each range array)

        nb1_ta = mpl_d['Background Average']
        delnb1s_ta = mpl_d['Background Std Dev']**2
        nb2_ta = mpl_d['Background Average 2']
        delnb2s_ta = mpl_d['Background Std Dev 2']**2

        ## updating mask
        r_trm = n_trm * (
            np.arange(r_tra.shape[1])
            >= (np.argmax(r_tra > BLINDRANGE, axis=1)[:, None])
        )


        # retrieve calibration files
        ## calc needed calibration files
        DeltNbinpad_ta = list(zip(Delt_ta, Nbin_ta, pad_ta))
        DeltNbinpad_a = list(set(DeltNbinpad_ta))
        napOE1_raa, napOE2_raa, delnapOE1s_raa, delnapOE2s_raa,\
            Oc_raa, delOcs_raa,\
            D_funca = np.apply_along_axis(
                _aaacaliprofiles_func, 0, np.array(DeltNbinpad_a).T,
                (lidarname, ), {
                    'mplreader': mplreader,
                    'plotboo': False,
                    'verbboo': True
                }
            )
        cali_raal = [napOE1_raa, napOE2_raa, delnapOE1s_raa, delnapOE2s_raa,
                     Oc_raa, delOcs_raa]
        D_func = D_funca[0]    # D_func's are all the same for the same lidar
        ## indexing calculated files
        DeltNbinpad_d = {DeltNbinpad: i
                         for i, DeltNbinpad in enumerate(DeltNbinpad_a)}
        DeltNbinpadind_ta = np.array(
            list(map(lambda x: DeltNbinpad_d[x], DeltNbinpad_ta)),
            dtype=np.int
        )
        napOE1_tra, napOE2_tra, delnapOE1s_tra, delnapOE2s_tra,\
            Oc_tra, delOcs_tra = [
                np.array(list(map(lambda x: raa[x],  DeltNbinpadind_ta)))
                for raa in cali_raal
            ]

        # change dtype of channels to cope for D_func calculation
        n1_tra = n1_tra.astype(np.float64)
        n2_tra = n2_tra.astype(np.float64)

        # pre calc derived quantities
        P1_tra = n1_tra * D_func(n1_tra)
        P2_tra = n2_tra * D_func(n2_tra)
        delP1s_tra = P1_tra/N_ta[:, None]
        delP2s_tra = P2_tra/N_ta[:, None]

        nap1_tra = napOE1_tra * E_ta[:, None]
        delnap1s_tra = (napOE1_tra * DELEOVERE * E_ta[:, None])**2\
            + (E_ta[:, None] * delnapOE1s_tra)**2
        nap2_tra = napOE2_tra * E_ta[:, None]
        delnap2s_tra = (napOE2_tra * DELEOVERE * E_ta[:, None])**2\
            + (E_ta[:, None] * delnapOE2s_tra)**2


        # compute NRB
        NRB1_tra = (
            (P1_tra - nb1_ta[:, None]) / E_ta[:, None]
            - napOE1_tra
        ) / Oc_tra * (r_tra**2)
        NRB2_tra = (
            (P2_tra - nb2_ta[:, None]) / E_ta[:, None]
            - napOE2_tra
        ) / Oc_tra * (r_tra**2)
        NRB_tra = NRB1_tra + NRB2_tra

        # compute delNRB
        delNRB1_tra = np.sqrt(
            (delP1s_tra + delnb1s_ta[:, None] + delnap1s_tra)
            / ((P1_tra - nb1_ta[:, None] - nap1_tra)**2)
            + DELEOVERE**2
            + delOcs_tra/(Oc_tra**2)
        )
        delNRB2_tra = np.sqrt(
            (delP2s_tra + delnb2s_ta[:, None] + delnap2s_tra)
            / ((P2_tra - nb2_ta[:, None] - nap2_tra)**2)
            + DELEOVERE**2
            + delOcs_tra/(Oc_tra**2)
        )
        delNRB_tra = np.sqrt(
            (
                delP1s_tra + delnb1s_ta[:, None] + delnap1s_tra
                + delP2s_tra + delnb2s_ta[:, None] + delnap2s_tra
            ) / ((
                P1_tra - nb1_ta[:, None] - nap1_tra
                + P2_tra - nb2_ta[:, None] - nap2_tra
            )**2)
            + DELEOVERE**2
            + delOcs_tra/(Oc_tra**2)
        )

        # handling nan values
        NRB1_tra = np.nan_to_num(NRB1_tra)
        NRB2_tra = np.nan_to_num(NRB2_tra)
        NRB_tra = np.nan_to_num(NRB_tra)
        delNRB1_tra = np.nan_to_num(delNRB1_tra)
        delNRB2_tra = np.nan_to_num(delNRB2_tra)
        delNRB_tra = np.nan_to_num(delNRB_tra)

        # handling altitude
        try:
            azi_ta = mpl_d['Azimuth Angle']
            ele_ta = mpl_d['Elevation Angle']

            # handling no scanner usage
            noscanscene_ta = ~(mpl_d['Scan Scenario Flag'].astype(np.bool))
            azi_ta[noscanscene_ta] = _static_azimuth
            ele_ta[noscanscene_ta] = _static_elevation

            theta_ta, phi_ta = LIDAR2SPHEREFN(np.stack([azi_ta, ele_ta], axis=1),
                                              np.deg2rad(ANGOFFSET))
            z_tra = np.cos(theta_ta)[:, None] * r_tra

            # creating theta set and index array
            theta_a = list(set(theta_ta))

            DeltNbinpadtheta_ta = list(map(
                tuple,
                np.append(DeltNbinpad_ta, theta_ta[:, None], axis=-1)
            ))
            DeltNbinpadtheta_a = list(set(DeltNbinpadtheta_ta))

            DeltNbinpadtheta_d = {
                DeltNbinpadtheta: i
                for i, DeltNbinpadtheta in enumerate(DeltNbinpadtheta_a)
            }
            DeltNbinpadthetaind_ta = np.array(list(map(
                lambda x: DeltNbinpadtheta_d[x],
                DeltNbinpadtheta_ta
            )))

        except KeyError:
            pass


        # Storing data
        ret_d = {
            'Timestamp': ts_ta,
            'DeltNbinpad_a': DeltNbinpad_a,
            'DeltNbinpadind_ta': DeltNbinpadind_ta,
            'r_tra': r_tra,
            'r_trm': r_trm,
            'NRB_tra': NRB_tra,
            'NRB1_tra': NRB1_tra,
            'NRB2_tra': NRB2_tra,
            'SNR_tra': NRB_tra/delNRB_tra,
            'SNR1_tra': NRB1_tra/delNRB2_tra,
            'SNR2_tra': NRB2_tra/delNRB2_tra,
            'delNRB1_tra': delNRB1_tra,
            'delNRB2_tra': delNRB2_tra,
            'delNRB_tra': delNRB_tra,
        }
        try:
            ret_d['z_tra'] = z_tra
            ret_d['theta_ta'] = theta_ta
            ret_d['theta_a'] = theta_a
            ret_d['phi_ta'] = phi_ta
            ret_d['DeltNbinpadtheta_a'] = DeltNbinpadtheta_a
            ret_d['DeltNbinpadthetaind_ta'] = DeltNbinpadthetaind_ta
        except NameError:
            pass


    # writing and reading from NRB file
        if writeboo:            # write to file
            ret_d = {key: ret_d[key].tolist() for key in list(ret_d.keys())}
            with open(DIRCONFN(SOLARISMPLDIR.format(lidarname),
                               DATEFMT.format(starttime),
                               NRBDIR.format(starttime, endtime)),
                      'w') as json_file:
                json_file.write(json.dumps(ret_d))

    else:                       # reading from file
        with open(DIRCONFN(SOLARISMPLDIR.format(lidarname),
                               DATEFMT.format(starttime),
                               NRBDIR.format(starttime, endtime)),
                  ) as json_file:
            ret_d = json.load(json_file)
        ret_d = {
            key: np.array(ret_d[key]) for key in list(ret_d.keys())
        }

    # performing time average
    if timestep:
        ret_d = time_average(ret_d, timestep)


    # resampling; if specified
    if rangestep:
        print(f'performing resampling with {rangestep=}')
        ret_d = range_resampler(ret_d, rangestep)


    # returning
    return ret_d


# testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from ...file_readwrite import smmpl_reader, mpl_reader

    lidarname = 'smmpl_E2'
    mplreader = smmpl_reader
    # mplfile_dir = DIRCONFN(osp.dirname(osp.abspath(__file__)),
    #                        'testNRB_smmpl_E2.mpl')
    mplfile_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200930/202009300734.mpl'
    starttime, endtime = None, None
    ret_d = main(
        lidarname, mplreader,
        mplfile_dir,
        starttime, endtime,
        genboo=True,
        writeboo=False
    )
    ts_ta = ret_d['Timestamp']
    z_tra = ret_d['z_tra']
    r_trm = ret_d['r_trm']
    NRB1_tra = ret_d['NRB1_tra']
    NRB2_tra = ret_d['NRB2_tra']
    NRB_tra = ret_d['NRB_tra']
    SNR1_tra = ret_d['SNR1_tra']
    SNR2_tra = ret_d['SNR2_tra']
    SNR_tra = ret_d['SNR_tra']

    # reading sigmaMPL data
    mplcsvfile_dir = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/20200930/202009300734_NRB.csv'
    sigmaNRB = pd.read_csv(mplcsvfile_dir, header=1, index_col=0).to_numpy().T
    polcut_ind = int((sigmaNRB.shape[1]+1)/2)
    sigmar_ra = pd.read_csv(mplcsvfile_dir, header=1)['Unnamed: 0'][:polcut_ind]
    sigmaNRB1_tra = sigmaNRB[:, :polcut_ind]
    sigmaNRB2_tra = sigmaNRB[:, polcut_ind:]
    sigmaNRB_tra = sigmaNRB1_tra + sigmaNRB2_tra

    # figure creating
    fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True)

    print('plotting the following timestamps:')
    for i in range(a := 11, a + 1):
        print(f'\t {ts_ta[i]}')

        # plotting computed data

        # ax.plot(z_tra[i][r_trm[i]], NRB1_tra[i][r_trm[i]], color='C0')
        # ax.plot(z_tra[i][r_trm[i]], NRB2_tra[i][r_trm[i]], color='C1')
        ax.plot(z_tra[i][r_trm[i]], NRB_tra[i][r_trm[i]], color='C2')
        # ax1.plot(z_tra[i][r_trm[i]], SNR1_tra[i][r_trm[i]], color='C0')
        # ax1.plot(z_tra[i][r_trm[i]], SNR2_tra[i][r_trm[i]], color='C1')
        ax1.plot(z_tra[i][r_trm[i]], SNR_tra[i][r_trm[i]], color='C2')
        # ax2.plot(
            # z_tra[i][r_trm[i]], ret_d['nb1_ta'][i]*np.ones_like(z_tra[i][r_trm[i]]),
        #     z_tra[i][r_trm[i]], ret_d['P1_tra'][i][r_trm[i]],
        #     color='C2'
        # )

        # plotting comparison with sigmaMPL
        ax.plot(sigmar_ra, sigmaNRB_tra[i], color='k')


    ax.set_yscale('log')
    ax1.set_yscale('log')
    # ax1.set_ylim([0, NOISEALTITUDE])
    plt.xlim([0, 15])
    plt.show()
