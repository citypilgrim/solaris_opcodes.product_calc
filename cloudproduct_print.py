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
    import os

    import numpy as np
    np.seterr(all='ignore')     # choosing not to print any warnings

    from .__main__ import main as product_calc
    # from .optimaltime_search import main as optimaltime_search
    from ..file_readwrite import smmpl_reader

    # mutable params
    lidarname = 'smmpl_E2'
    combpol_boo = True
    pixelsize = 5               # [km]
    gridlen = 3

    angularoffset = 140.6                      # [deg]
    latitude, longitude = 1.299119, 103.771232  # [deg]
    elevation = 70                              # [m]


    # retreiving optimal time
    # starttime, endtime = optimaltime_search(LOCTIMEFN(datetime.now(), UTCINFO))
    starttime = LOCTIMEFN('202011250000', UTCINFO)
    endtime = LOCTIMEFN('202011251200', UTCINFO)

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
