# imports
import numpy as np
import pandas as pd

from ..file_readwrite import smmpl_reader
from ..global_imports.solaris_opcodes import *

# params

_optimal_timdelta = pd.Timedelta(OPTIMALTIMEDELTA, 'h')
_timestamp_key = 'Timestamp'
_azimuth_key = 'Azimuth Angle'
_elevation_key = 'Elevation Angle'


# main func
@verbose
@announcer
def main(
        lidarname,
        starttime=None,
        endtime=None,
):
    '''
    searches for the optimal start/end time for a given end/start time, depending
    on which is provided.
    an optimal time is defined as the latest starttime such that the data is a
    complete sweep of the atmosphere.

    The time delta of an optimal time sweep from start to end should be less than or
    equal to OPTIMALTIMEDELTA. If not it will return the last value in optimal time
    delta

    Parameters
        lidarname (str): directory name of lidar
        start/endtime (datetime like): the approx start/end time from which we want to
                                       perform the optimal time search
    Return
        end/starttime (datetime like): corresponding end/start time for the given
                                       parameter
    '''
    # determining search backward or search forward
    if starttime:
        endtime = starttime + _optimal_timdelta
        searchbackward_boo = False
    else:
        starttime = endtime - _optimal_timdelta
        searchbackward_boo = True

    # reading data
    mpl_d = smmpl_reader(
        DIRCONFN(SOLARISMPLDIR.format(lidarname)),
        starttime=starttime, endtime=endtime,
    )
    ts_ta = mpl_d[_timestamp_key]
    azi_ta = mpl_d[_azimuth_key]
    ele_ta = mpl_d[_elevation_key]

    # finding next point in time the same angle occurs
    if searchbackward_boo:
        searchazi = azi_ta[-1]
        searchele = ele_ta[-1]
        azi_ta = azi_ta[:-1][::-1]
        ele_ta = ele_ta[:-1][::-1]
    else:
        searchazi = azi_ta[0]
        searchele = ele_ta[0]
        azi_ta = azi_ta[1:]
        ele_ta = ele_ta[1:]

    sameang_boo = (azi_ta == searchazi) * (ele_ta == searchele)
    if sameang_boo.any():
        # 1 accounts for the removed search value
        searchind = np.argmax(sameang_boo) + 1
        if searchbackward_boo:
            searchind *= -1
    else:
        if searchbackward_boo:
            searchind = 0
        else:
            searchind = -1

    # returning the right timestamp
    return ts_ta[searchind]


# testing
if __name__ == '__main__':
    # endtime = main(
    starttime = main(
        'smmpl_E2',
        # starttime=LOCTIMEFN('202011250000', 0)
        endtime=LOCTIMEFN('202011252300', 0)
    )
    # print(endtime)
    print(starttime)
