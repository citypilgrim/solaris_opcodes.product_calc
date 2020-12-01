# imports
from copy import deepcopy

import numpy as np


# main func
def main(
        prodmask_ggAl, gridlen
):
    '''
    Interpolating across pixels for empty pixels, for a given empty pixel,
    we shall take the average lowest layer from the nearest neighbouring pixels

    promask_ggAl (np.ndarray): product mask altitudes
                               shape: (gridlen, gridlen, cloud layers)
    gridlen (int): size of grids; number of pixels along one side of the grid
    '''
    oprodbot_ggAl = deepcopy(prodmask_ggAl)
    for i in range(gridlen):
        for j in range(gridlen):
            if not prodbot_ggAl[i][j]:

                # find all neighbouring indexes
                neighbouri_a = np.array([i-1, i-1, i-1, i, i, i+1, i+1, i+1])
                neighbourj_a = np.array([j-1, j, j+1, j-1, j+1, j-1, j, j+1])

                # filtering invalid neighbouring indexes



                # taking averaging of neighbouring pixels


    return prodmask_ggAl
