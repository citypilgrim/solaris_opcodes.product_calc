# imports
import numpy as np


# main func
def main(
        prodmask_ta
):
    '''
    Converts product mask which is an array containing a list of [prod_bot, prod_top],
    with each list corresponding to a layer. Into stacking of two arrays, one for
    prod_bot and the other for prod_top
    with the shape (time, max number of layers, 2(cldbot, cldtop))
    with height is not found, will be left as np.nan

    Parameters
        prodmask_ta (np.ndarray): product mask array, for example, see return of
                                  product_calc.cloud_calc.gcdm
    Return
        prodmask_tl2a (np.ndarray): product mask, np.nan if not found
    '''
    # initialise arrays
    maxlayer = max([prodmask.shape[0] for prodmask in prodmask_ta])
    prodtop_tla = np.empty((len(prodmask_ta), maxlayer))
    prodtop_tla[:] = np.nan
    prodbot_tla = np.empty((len(prodmask_ta), maxlayer))
    prodbot_tla[:] = np.nan

    # filling in arrays
    for i, prodmask in enumerate(prodmask_ta):
        for j, mask in enumerate(prodmask):
            prodbot_tla[i, j], prodtop_tla[i, j] = mask

    return np.stack([prodbot_tla, prodtop_tla], axis=-1)



# testing
if __name__ == '__main__':
    main()
