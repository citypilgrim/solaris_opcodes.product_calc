# imports
from ..global_imports.solaris_opcodes import *


# main func
@verbose
def main(
        product_d,
):
    '''
    this function serves to retrieve the computed cloud products from
    product_calc.__main__

    Parameters
        product_d (dict): output from product_calc.__main__
    '''
    ts_ta = product_d[NRBKEY]['Timestamp']
    print(ts_ta[0])
    print(ts_ta[-1])

    lat_pa = product_d[PIXELLATITUDEKEY]
    long_pa = product_d[PIXELLONGITUDEKEY]
    cloud_d = product_d[CLOUDKEY]

    cloudbottom_pAl = cloud_d[PIXELBOTTOMKEY]

    for i, lat in enumerate(lat_pa):
        print(lat_pa[i], long_pa[i], *cloudbottom_pAl[i])


# running
if __name__ == '__main__':
    '''
    The operational usage of this code is to print out the cloud bottoms for the
    corresponding grid.

    starttime will be the time the program is run, and it would retreieve the latest
    product which would span the entire sky

    USER must adjust mutable aprams according to their needs
    '''
    # imports
    import datetime as dt
    import os

    import numpy as np
    np.seterr(all='ignore')     # choosing not to print any warnings

    from .__main__ import main as product_calc
    from .optimaltime_search import main as optimaltime_search
    from ..file_readwrite import smmpl_reader
    from ..lidardata_pull import main as lidardata_pull

    # mutable params
    lidarname = 'smmpl_E2'

    lidar_ip = None
    lidaruser = 'mpluser'
    lidardata_dir = f'C:/Users/mpluser/Desktop/{lidarname}'

    combpol_boo = True
    pixelsize = 5               # [km]
    gridlen = 3

    angularoffset = 140.6                      # [deg]
    latitude, longitude = 1.299119, 103.771232  # [deg]
    elevation = 70                              # [m]


    # pulling latest dataset from the lidar
    lidardata_pull(
        lidar_ip, lidaruser,
        lidardata_dir,
        verbboo=False
    )


    # retreiving optimal time
    starttime = LOCTIMEFN(dt.datetime.now(), 0)
    endtime = optimaltime_search(
        lidarname,
        starttime=starttime,
        verbboo=False,
    )

    # running computation
    product_d = product_calc(
        lidarname, smmpl_reader,
        starttime=starttime, endtime=endtime,
        angularoffset=angularoffset,

        combpolboo=combpol_boo,

        pixelsize=pixelsize, gridlen=gridlen,
        latitude=latitude, longitude=longitude, elevation=elevation,

        verbboo=False,
    )

    # printing output
    main(product_d)
    os._exit(0)  # closes all the child processes that were left hanging
