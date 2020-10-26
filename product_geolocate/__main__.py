# imports
from .maskbottom_locate import main as maskbottom_locate


# main func
def main(
        product_d,
        pixelsize, gridlen,
        latitude, longitude, elevation
):
    '''
    geolocates the products provided to a map like grid. It augments the data in
    product_d with the geolocated data.

    Parameters
        product_d (dict): output from product_calc.__main__
        pixelsize (float): [km] pixel size for data sampling
        gridlen (int): length of map grid in consideration for geolocation
        latitude (float): [deg] coords of lidar
        longitude (float): [deg] coords of lidar
        elevation (float): [m] height of lidar from MSL
    '''

    # computing grid boundaries

    # finding bottom for cloud mask



# testing
if __name__ == '__main__':
    main()
