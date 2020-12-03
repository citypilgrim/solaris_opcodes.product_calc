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

    In principle this function could be improved upon to perform interpolation when
    the layer counts are very weak for a given pixel. But I have a feeling this is
    always true for a single sweep through the scan pattern

    promask_ggAl (list): product mask altitudes
                         shape: (gridlen, gridlen, cloud layers)
    gridlen (int): size of grids; number of pixels along one side of the grid
    '''
    oprodmask_ggAl = deepcopy(prodmask_ggAl)  # averaging will always refer to original
    for i in range(gridlen):
        for j in range(gridlen):
            if not oprodmask_ggAl[i][j].size:

                # find all neighbouring indexes
                neighbouri_a = np.array([i-1, i-1, i-1, i, i, i+1, i+1, i+1])
                neighbourj_a = np.array([j-1, j, j+1, j-1, j+1, j-1, j, j+1])

                # filtering invalid neighbouring indexes
                remove_m = (
                    (neighbouri_a >= 0) * (neighbourj_a >= 0)
                    * (neighbouri_a < gridlen) * (neighbourj_a < gridlen)
                ).astype(np.bool)
                neighbouri_a = neighbouri_a[remove_m]
                neighbourj_a = neighbourj_a[remove_m]

                # taking averaging of lowest layer in neighbouring pixels
                # the slicing to index 1 instead of indexing at 0 prevents throwing
                # of exception in the event the neighbouring pixel is also empty
                prodmaskcandidate_a = np.array([])
                for k, neighbouri in enumerate(neighbouri_a):
                    neighbourj = neighbourj_a[k]
                    prodmaskcandidate_a = np.concatenate([
                        prodmaskcandidate_a,
                        oprodmask_ggAl[neighbouri][neighbourj][:1]
                    ])
                prodmask_ggAl[i][j] = np.array([
                    prodmaskcandidate_a.sum()/prodmaskcandidate_a.size
                ])

    return prodmask_ggAl

# testing
if __name__ == '__main__':

    # initialising a 3x3 grid, with some empty pixels
    prodmask_ggAl = [
        [
            [1],
            [],
            [2, 1],
        ],
        [
            [1],
            [],
            [1],
        ],
        [
            [1],
            [],
            [1],
        ],
    ]
    gridlen = 3

    prodmask_ggAl = main(prodmask_ggAl, gridlen)

    print(prodmask_ggAl)
