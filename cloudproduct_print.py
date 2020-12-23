# imports
from ..global_imports.solaris_opcodes import *


# main func
@logger
def main(
        product_d,
):
    '''
    this function serves to retrieve the computed cloud products from
    product_calc.__main__

    Parameters
        product_d (dict): output from product_calc.__main__
    '''
    if not product_d:
        print('no data found')

    else:
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

    USER must adjust mutable params according to their needs
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
    from ..global_imports.smmpl_opcodes import LIDARIPADDRESS

    # mutable params
    lidarname = 'smmpl_E2'

    lidar_ip = LIDARIPADDRESS
    lidaruser = 'mpluser'
    lidardata_dir = f'C:/Users/mpluser/Desktop/{lidarname}'

    lidar_utcoffset = 0         # [hrs]

    combpol_boo = True
    pixelsize = 5               # [km]
    gridlen = 3

    angularoffset = 140.6                      # [deg]
    latitude, longitude = 1.299119, 103.771232  # [deg]
    elevation = 70                              # [m]

    # pulling latest dataset from the lidar
    utcoffset = SOLARISUTCOFFSET - lidar_utcoffset
    today = dt.datetime.today() - dt.timedelta(hours=utcoffset)
    yesterday = today - dt.timedelta(days=1)
    lidardata_pull(
        lidar_ip, lidaruser,
        lidardata_dir,
        lidarname,
        syncday_lst=[
            DATEFMT.format(today),
            DATEFMT.format(yesterday)
        ],
        verbboo=False
    )

    # retreiving optimal time
    endtime = LOCTIMEFN(dt.datetime.now(), SOLARISUTCOFFSET)
    starttime = optimaltime_search(
        lidarname,
        endtime=endtime,
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

    # writing output to file
    log_file = DIRCONFN(
        SOLARISCLOUDPRODDIR.format(lidarname), DATEFMT.format(today),
        CLOUDPRODUCTFILE
    )
    main(product_d, stdoutlog=log_file, stderrlog=log_file, ,
         dailylogboo=True, utcoffset=utcoffset)

    # closes all the child processes that were left hanging
    os._exit(0)
