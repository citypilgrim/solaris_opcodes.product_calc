# imports
from ..global_imports.solaris_opcodes import *

# main func
@verbose
@announcer
def main(
        product_d
):
    '''
    generates formatted cloud products from the product_calc.__main__

    Parameters
        product_d (dict): output from product_calc.__main__
    '''
    lat_a = product_d[PRODUCTLATKEY]
    long_a = product_d[LONGITUDE]
    cloud_d = product_d[CLOUDKEY]

    cloudbottom_a = cloud_d[CLOUDBOTTOMKEY]

    for i, lat in enumerate(lat_a):
        print(lat_a[i], long_a[i], cloudbottom_a[i])


# running
if __name__ == '__main__':
    '''
    The operational usage of this code is to print out the cloud bottoms for the
    corresponding grid.

    starttime will be the time the program is run, and it would retreieve the latest
    product which would span the entire sky

    USER must adjust mutable aprams according to their needs
    '''
    import datetime.datetime as datetime

    from .__main__ import main as product_calc
    from .optimaltime_search import main as optimaltime_search
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
    starttime, endtime = optimaltime_search(LOCTIMEFN(datetime.now(), UTCINFO))

    # running computation
    product_d = product_calc(
        lidarname, smmpl_reader,
        starttime, endtime,
        angularoffset=angularoffset,

        combpolboo=combpol_boo,

        pixelsize=pixelsize, gridlen=gridlen,
        latitude=latitude, longitude=longitude, elevation=elevation,

        verbboo=False,
    )

    # printing output
    main(product_d)
