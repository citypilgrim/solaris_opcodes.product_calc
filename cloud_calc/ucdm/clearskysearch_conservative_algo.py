# imports
import numpy as np


# main func
def main(
        CRprime_ra, delCRprime_ra, N_ra, z_ra
):
    '''
    Does the clear sky search for a given profile. Stops the search once the
    number of significant bins has exceeded the remaining length of the array

    Any restriction on the minimum or maximum heights of the clear sky search has
    to be done on the arguments before calling the function

    return values are np.nan if clear sky cannot be found

    Parameters
        CRprime_ra (np.array): NRB corrected by molecular profile, 1d in altitude
        delCRprime_ra (np.array): uncertainty of th eabove, 1d in altitude
        N_ra (np.array): number of significant bins required, 1d in altitude
        z_ra (np.array): altitude array

    Return
        Cfstar (float): the calibration constant which is approximately equal
                        to the CTp2(z=r_N), r_N is the height of thr average
        delCfstar (float): uncertainty (given by stdev) of the above
        lowalt (float): lower height boundary of clear sky region found
        highalt (float): higher height boundary of clear sky region found
    '''
    CRprime_a = CRprime_ra
    delCRprime_a = delCRprime_ra
    for i, CRprime in enumerate(CRprime_ra):
        delCRprime = delCRprime_ra[i:]
        N = N_ra[i]
        CRprime_a = CRprime[1:]
        delCRprime_a = delCRprime_a[1:]

        if N > CRprime_a.size:
            # num of significant bins has exceeded the array slice, indicating
            # we have reached the maximum altitude possible for the clear sky
            # search
            return [np.nan, np.nan, np.nan, np.nan]

        else:

            boo_a = (CRprime_a + delCRprime_a >= CRprime - delCRprime)\
                * (CRprime_a - delCRprime_a <= CRprime + delCRprime)

            if boo_a[:N].all():  # passed the clear sky requirement
                Cfstar = CRprime_a[:N].mean()
                delCfstar = CRprime_a[:N].stdev()
                lowalt = z_ra[i+1]
                highalt = z_ra[i+N]

                return np.array([Cfstar, delCfstar, lowalt, highalt])


# testing
if __name__ == '__main__':
    main()
