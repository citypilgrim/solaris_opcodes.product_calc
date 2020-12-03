# imports
import numpy as np


# main func
def main(
        prodmask_ta
):
    '''
    Converts product mask which is an array containing a list of
    [prod_bot, prod_peak, prod_top],
    with each list corresponding to a layer. Into stacking of 3 arrays, one for
    prod_bot, prod_peak and prod_top each.
    with the shape (time, max number of layers, 3(cldbot, ,cldpeak, cldtop))
    with height is not found, will be left as np.nan

    Parameters
        prodmask_ta (np.ndarray): product mask array, for example, see return of
                                  product_calc.cloud_calc.gcdm
    Return
        prodmask_tl3a (np.ndarray): product mask, np.nan if not found
    '''
    # initialise arrays
    maxlayer = max([prodmask.shape[0] for prodmask in prodmask_ta])
    prodbot_tla = np.empty((len(prodmask_ta), maxlayer))
    prodbot_tla[:] = np.nan
    prodpeak_tla = np.empty((len(prodmask_ta), maxlayer))
    prodpeak_tla[:] = np.nan
    prodtop_tla = np.empty((len(prodmask_ta), maxlayer))
    prodtop_tla[:] = np.nan


    # filling in arrays
    for i, prodmask in enumerate(prodmask_ta):
        for j, mask in enumerate(prodmask):
            prodbot_tla[i, j], prodpeak_tla[i, j], prodtop_tla[i, j] = mask

    return np.stack([prodbot_tla, prodpeak_tla, prodtop_tla], axis=-1)



# testing
if __name__ == '__main__':
    main()
