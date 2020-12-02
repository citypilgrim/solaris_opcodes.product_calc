# imports
import numpy as np

from ...global_imports.solaris_opcodes import *


# params
_histogram_range = (BOTTOMBLINDRANGE, TOPBLINDRANGE)
_histogram_binnum = int(np.ceil(np.diff(_histogram_range)[0]/HISTOGRAMBINSIZE))
_histogram_gapthreshold = HISTOGRAMGAPSIZE

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
        histavg_a (np.ndarray): averaged mask altitude for each layer,
                                len == 1 if peakonly_boo
    '''
    # splitting into histogram bins
    histlen_ba, binedges_ba = np.histogram(mask_a,
                                           bins=_histogram_binnum,
                                           range=_histogram_range)
    ## getting indexes for elements to their own bins
    inds = np.digitize(mask_a, binedges_ba)
    ## splitting array into their bins
    hist_bAl = [mask_a[inds == i+1] for i in range(_histogram_binnum)]

    # determining layers from histogram bin proximity
    histlen_ll, hist_lAl = [], []
    histlen_l, hist_a = [], np.array([])
    empty_count = _histogram_gapthreshold
    for i, histlen in enumerate(histlen_ba):
        if not histlen:
            empty_count -= 1
            if empty_count <= 0:
                if histlen_l:
                    hist_lAl.append(hist_a)
                    histlen_ll.append(histlen_l)
                    histlen_l, hist_a = [], np.array([])
        else:
            empty_count = _histogram_gapthreshold
            hist_a = np.concatenate([hist_a, hist_bAl[i]], axis=-1)
            histlen_l.append(histlen_ba[i])
    if histlen_l:   # handling last layer, incase no gap at the end
        hist_lAl.append(hist_a)
        histlen_ll.append(histlen_l)

    # taking average
    masklen_l = [sum(histlen_l) for histlen_l in histlen_ll]
    maskavg_l = [sum(hist_a)/masklen_l[i] if masklen_l[i] else 0
                 for i, hist_a in enumerate(hist_lAl)]

    if peakonly_boo:
        # if we only want the strongest cloud, will return the height of the
        # layer with the highest count
        maxlen = max(masklen_l)
        return np.array([maskavg for i, maskavg in enumerate(maskavg_l)
                         if masklen_l[i] == maxlen][:1])
    else:
        return np.array(maskavg_l)


# testing
if __name__ == '__main__':
    pass
