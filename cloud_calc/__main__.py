# imports
from .gcdm import main as gcdm
# from .ucdm import main as ucdm

# main func
def main(
        nrbdic,
        smmplboo=True, pixelsize=None,
        combpolboo=True,
        plotboo=False,
):
    '''
    Produces cloud masks and the corresponding cloud information
    Currently ucdm is not in use.

    Parameters
        nrbdic (dict): output from .nrb_calc.py
        smmplboo (boolean): decides whether or not it is necesssary average data
                            in grid points
        pixelsize (float): [km], averaging size of pixel
        combpolboo (boolean): gcdm on combined polarizations or just co pol
        plotboo (boolean): whether or not to plot computed results
    Return
        cloudmask_ta (np.ndarray): each timestamp contains a list of tuples for the
                                   clouds
        cloudbottom_a (np.ndarray):
    '''



# testing
if __name__ == '__main__':
    main()
