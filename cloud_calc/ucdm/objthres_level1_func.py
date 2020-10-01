# imports
import numpy as np


# main func
def main(
        delCfstar_ta, Cfstar_ta,
        betamprime_tra,
        delNRB_tra,
):
    '''
    params are as specified in equation 9 of Lewis et. al 2016
    '''
    return (
        betamprime_tra
        + betamprime_tra * np.sqrt(
            # (
            #     delNRB_tra/(betamprime_tra * Cfstar_ta[:, None])
            # )**2
            + (delCfstar_ta/Cfstar_ta)[:, None]**2
        )
    )


# testing
if __name__ == '__main__':
    main()
