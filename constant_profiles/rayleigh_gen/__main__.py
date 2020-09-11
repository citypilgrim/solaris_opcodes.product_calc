# imports
import os.path as osp

import netCDF4 as nc
import numpy as np
from scipy.integrate import cumtrapz

from ....global_imports.solaris_opcodes import *


# static params
_mollidrat = np.pi*8/3   # molecular lidar ratio


# main func
@verbose
@announcer
def main(
        arg_a,
        # Delt, Nbin, theta=None,
        weather=WEATHER, wavelength=WAVELENGTH
):
    '''
    function that reads rayleigh-523_sing.cdf, stores data for the stated lambda

    Parameters
        arg_a (array like): contains the floowing
            Delt (float): bintime
            Nbin (int): number of bins
            (optional) theta (float): [rad] lidar to zenith angle
        wavelength (float): [nm], wavelength of light used
        weather (str): 'winter' or 'summer'

    Return
        betam_ra (np.array): molecular backscattering w.r.t altitude
        T2m_ra (np.array): molecular transmission squared w.r.t altitude
    '''
    print(arg_a)
    try:
        Delt, Nbin, theta = arg_a
    except IndexError:
        Delt, Nbin = arg_a
        theta = 0

    # reading scattering coefficient file
    ray_file = DIRCONFN(osp.dirname(osp.abspath(__file__)),
                        RAYLEIGHCDFDIR.format(RAYLEIGHCDLAMBDA))
    rayscatdat_nc = nc.Dataset(ray_file, 'r', format='NETNC4_CLASSIC')
    ncr_ra = rayscatdat_nc.variables['range'][:]
    betam_ra = rayscatdat_nc.variables[weather+'_ray'][:]

    # interpolating
    Delz = Delt * SPEEDOFLIGHT * np.cos(theta)
    z_ra = Delz * np.arange(Nbin) + Delz/2
    betam_ra = np.interp(z_ra, ncr_ra, betam_ra)

    # computing back scatter
    betam_ra *= ((wavelength/RAYLEIGHCDLAMBDA)**(-4))

    # computing transmission
    sigmam_ra = _mollidrat * betam_ra              # scat cross sec
    Tm2_ra = np.exp(-2 * cumtrapz(sigmam_ra, z_ra, initial=0))

    # computing product
    betamprime_ra = betam_ra * Tm2_ra

    # returning
    return betam_ra, Tm2_ra, betamprime_ra


# generating
if __name__ == '__main__':
    Delt = 1e-7
    Nbin = 1200
    Delr = Delt * SPEEDOFLIGHT
    r_ra = Delr * np.arange(Nbin) + Delr/2
    betam_ra, T2m_ra = main(True, Delt, Nbin)
