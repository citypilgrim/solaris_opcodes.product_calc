# imports
import numpy as np

from ...global_imports.solaris_opcodes import *


# params
_histogram_range = (BOTTOMBLINDRANGE, TOPBLINDRANGE)
_histogram_binnum = np.diff(_histogram_range)[0]/HISTOGRAMBINSIZE


# main func
def main(
        mask_a, peakonly_boo=True,
):
    '''
    From a one dimensional array of mask altitude, will organise them
    into histogram bins of fixed sizes.
    Then it will catergorise bins into their respective layers,
    by finding the gaps between bins.
    And finally compute the average altitude for each layer

    The layer search is not optimized because the number of bins
    are not meant to be that many ~ 30 is reasonable.

    Parameters
        mask_a (np.ndarray): 1D array of mask altitudes to be binned
        peakonly_boo (boolean): decides whether or not to return the
                                average for each layer or just the
                                layer with the most number of points
    Return
        maskavg_a (np.ndarray): averaged mask altitude for each layer,
                                len == 1 if peakonly_boo
    '''
    # splitting into histogram bins
    # 'A' represents variable length
    hist_bAt, binedges_ba = np.histogram(mask_a,
                                       bins=_histogram_binnum,
                                       range=_histogram_range)
    histlen_ba = np.array([len(hist_At) for hist_At in hist_bAt])

    # determining layers from histogram bin proximity
    histempty_bm = (histlen_bt == 0)
    histlen_lAl, hist_lAl = [], []
    histlen_Al, hist_Al = [], []
    for i, b in enumerate(histempty_bm):
        '''CONTINUE HERE'''

    # taking average


# testing
if __name__ == '__main__':
    main()
