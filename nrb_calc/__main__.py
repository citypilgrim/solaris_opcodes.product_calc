# imports
import json
import os.path as osp

import numpy as np

from ..cali_profiles import cali_profiles
from ...file_readwrite import mpl_reader, smmpl_reader
from ...global_imports.solaris_opcodes import *


# supp func
def _aaacaliprofiles_func(dim1arr, args, kwargs):
    '''
    Parameters
        args (tuple): (lidarname,)
        kwargs (dict)
    '''
    return cali_profiles(*args, *dim1arr, **kwargs)

def _D_func(D_func, n_ara):
    return D_func(n_ara)
_vecD_func = np.vectorize(_D_func)


# main func
@verbose
@announcer
def main(
        lidarname, mplreader,
        mplfiledir=None,
        starttime=None, endtime=None,
        genboo=True,
        writeboo=False,
):
    '''
    uncert in E ~= 1% assuming that measurement averages are typically <= 1min,
    which is equivalent to temperature fluctuations of <= 2% according to
    campbell 20002 uncertainty paper

    Parameters
        lidarname (str): directory name of lidar
        mplfiledir (str): mplfile to be processed if specified, date should be
                          None
        start/endtime (datetime like): approx start/end time of data of interest
        genboo (boolean): if True, will read .mpl files and generate NRB, return
                           and write
                           if False, will read exisitng nrb files and only return
        mplreader (func): either mpl_reader or smmpl_reader,
                          must be specified if genboo is True
        writeboo (boolean): data is written to filename if True,
                             ignored if genboo is False

    Return
        ret_d (dict):
            DeltNbin_a (list): set of zipped Delt_ta and Nbin_ta
            DeltNbinind_ta (np.array): shape (time dim), tra index on set of Nbin
                                       and Delt
            r_tra (np.array): shape (time dim, no. range bins)
            r_trm (np.array): shape (time dim, no. range bins), usually all ones
            NRB1/2_tra (np.array): shape (time dim, no. range bins)
            delNRB1/2_tra (np.array): shape (time dim, no. range bins)
            SNR1/2_tra (np.array): shape (time dim, no. range bins)
            theta/phi_ta (np.array): [rad] if smmpl_boo,
                                     spherical coordinates of data, angular offset
                                     corrected
    '''
    # checking which lidar we are dealing with
    smmpl_boo = (mplreader is smmpl_reader)


    # computation
    if genboo:
        # read .mpl files
        mpl_d = mplreader(
            lidarname,
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
        Nbin_ta = mpl_d['Number Bins']

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
        DeltNbin_ta = list(zip(Delt_ta, Nbin_ta))
        DeltNbin_a = list(set(DeltNbin_ta))
        napOE1_raa, napOE2_raa, delnapOE1s_raa, delnapOE2s_raa,\
            Oc_raa, delOcs_raa,\
            D_funca = np.apply_along_axis(
                _aaacaliprofiles_func, 0, np.array(DeltNbin_a).T,
                (lidarname, ), {'genboo':True, 'verbboo':True}
            )
        cali_raal = [napOE1_raa, napOE2_raa, delnapOE1s_raa, delnapOE2s_raa,
                     Oc_raa, delOcs_raa]
        D_func = D_funca[0]    # D_func's are all the same for the same lidar
        ## indexing calculated files
        DeltNbin_d = {DeltNbin:i for i, DeltNbin in enumerate(DeltNbin_a)}
        DeltNbinind_ta = np.array(list(map(lambda x: DeltNbin_d[x],
                                           DeltNbin_ta)))
        napOE1_tra, napOE2_tra, delnapOE1s_tra, delnapOE2s_tra,\
            Oc_tra, delOcs_tra = [
                np.array(list(map(lambda x:raa[x],  DeltNbinind_ta)))
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


        # Storing data
        ret_d = {
            'Timestamp': ts_ta,
            'DeltNbin_a': DeltNbin_a,
            'DeltNbinind_ta': DeltNbinind_ta,
            'r_tra': r_tra,
            'r_trm': r_trm,
            'NRB_tra': NRB_tra,
            'NRB1_tra': NRB1_tra,
            'NRB2_tra': NRB2_tra,
            'SNR_tra': NRB_tra/delNRB_tra,
            'SNR1_tra': NRB1_tra/delNRB2_tra,
            'SNR2_tra': NRB2_tra/delNRB2_tra,
        }
        if smmpl_boo:
            theta_ta, phi_ta = LIDAR2SPHEREFN(np.stack(
                [mpl_d['Azimuth Angle'], mpl_d['Elevation Angle']], axis=1
            ), np.deg2rad(ANGOFFSET))
            ret_d['theta_ta'] = theta_ta
            ret_d['phi_ta'] = phi_ta


        # writing to file
        if writeboo:
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
            key:np.array(ret_d[key]) for key in list(ret_d.keys())
        }

    # returning
    return ret_d


# testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ...file_readwrite import mpl_reader

    lidarname = 'mpl_S2S'
    mplfile_dir = DIRCONFN(osp.dirname(osp.abspath(__file__)),
                           'testNRB_mpl_S2S.mpl')
    starttime, endtime = None, None
    ret_d = main(
        lidarname, mpl_reader,
        mplfile_dir,
        starttime, endtime,
        genboo=True,
        writeboo=False
    )

    fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True)
    ts_ta = ret_d['Timestamp']
    r_tra = ret_d['r_tra']
    r_trm = ret_d['r_trm']
    NRB1_tra = ret_d['NRB1_tra']
    NRB2_tra = ret_d['NRB2_tra']
    NRB_tra = ret_d['NRB_tra']
    SNR1_tra = ret_d['SNR1_tra']
    SNR2_tra = ret_d['SNR2_tra']
    SNR_tra = ret_d['SNR_tra']

    print('plotting the following timestamps:')
    for i in range(a:=300, a+10):
        print(f'\t {ts_ta[i]}')
        ax.plot(r_tra[i][r_trm[i]], NRB1_tra[i][r_trm[i]], color='C0')
        ax.plot(r_tra[i][r_trm[i]], NRB2_tra[i][r_trm[i]], color='C1')
        ax.plot(r_tra[i][r_trm[i]], NRB_tra[i][r_trm[i]], color='C2')
        ax1.plot(r_tra[i][r_trm[i]], SNR1_tra[i][r_trm[i]], color='C0')
        ax1.plot(r_tra[i][r_trm[i]], SNR2_tra[i][r_trm[i]], color='C1')
        ax1.plot(r_tra[i][r_trm[i]], SNR_tra[i][r_trm[i]], color='C2')

    # ax.set_yscale('log')
    # ax1.set_yscale('log')
    ax1.set_ylim([0, NOISEALTITUDE])
    plt.xlim([0, 20])
    plt.show()
