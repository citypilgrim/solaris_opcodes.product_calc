# imports
import numpy as np
import pandas as pd

from ...global_imports.solaris_opcodes import *


# main func
def main(
        mplreader, mplfiledir, Dfunc,
        napOEraa,
        plotboo=False,
        slicetup=slice(AFTERPULSEPROFSTART, AFTERPULSEPROFEND, 1),
        compstr='a'
):
    r_ra, Oc_ra = pd.read_csv(
        mplfiledir,
        header=OVERLAPCSVHEADER
    ).to_numpy().T

    # computing uncertainty
    ## scaling
    delOc_ra = OVERLAPUNCERTSCALE * Oc_ra
    ## trimming uncertainty
    for i, _ in enumerate(Oc_ra):
        if list(Oc_ra[i:i+5]) == [1, 1, 1, 1, 1]:
            break

    delOc_ra = np.concatenate((delOc_ra[:i], np.zeros(len(r_ra)-i)))

    return r_ra, Oc_ra, delOc_ra


if __name__ == '__main__':
    # imports
    import matplotlib.pyplot as plt

    # testing
    csv_fn = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910230900_2e-7overlap.csv'

    r_ra, Oc_ra, delOc_ra = main(
        None, None, csv_fn, None, None
    )

    # checking to see if the scale factor is correct
    fig, ax = plt.subplots()
    ax.plot(r_ra, Oc_ra)
    ax.plot(r_ra, delOc_ra)

    ax.set_yscale('log')
    plt.show()
