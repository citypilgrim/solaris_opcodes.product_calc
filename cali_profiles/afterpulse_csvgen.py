# imports
import numpy as np
import pandas as pd

from ...global_imports.solaris_opcodes import *


# main func
def main(
        mplreader, lidarname, mplfiledir, Dfunc,
        plotboo=False,
        slicetup=slice(AFTERPULSEPROFSTART, AFTERPULSEPROFEND, 1),
        compstr='a'
):
    r_ra, napOE2_ra, napOE1_ra = pd.read_csv(
        mplfiledir,
        header=AFTERPULSECSVHEADER
    ).to_numpy().T

    # computing uncertainty by scaling
    delnapOE1_ra = AFTERPULSEUNCERTSCALE * napOE1_ra
    delnapOE2_ra = AFTERPULSEUNCERTSCALE * napOE2_ra

    return r_ra, napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra


if __name__ == '__main__':
    # imports
    import matplotlib.pyplot as plt

    # testing
    csv_fn = '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/201910170400_2e-7afterpulse.csv'

    r_ra, napOE1_ra, napOE2_ra, delnapOE1_ra, delnapOE2_ra = main(
        None, None, csv_fn, None
    )

    # checking to see if the scale factor is correct
    fig, ax = plt.subplots()
    ax.plot(r_ra, napOE1_ra)
    ax.plot(r_ra, napOE2_ra)
    ax.plot(r_ra, delnapOE1_ra)
    ax.plot(r_ra, delnapOE2_ra)

    ax.set_yscale('log')
    plt.show()
