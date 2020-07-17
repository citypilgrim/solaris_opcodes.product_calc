# imports
import netCDF4 as nc
import numpy as np
from scipy.integrate import cumtrapz

from ....globalimports import *


# static params
_mollidrat = np.pi*8/3   # molecular lidar ratio


# main func
@verbose
@announcer
def main(
        Delt, Nbin,
        genboo=True,
        weather=WEATHER, wavelength=WAVELENGTH
):
    '''
    function that reads rayleigh-523_sing.cdf, stores data for the stated lambda

    Future
        - convert to generate and read function, default with generating

    Parameters
        Delt (float): bintime
        Nbin (int): number of bins
        genboo (boolean): decides whether or not to generate profile or just read
        wavelength (float): [nm], wavelength of light used
        weather (str): 'winter' or 'summer'

    Return
        betam_ra (np.array): molecular backscattering as function of range
        T2m_ra (np.array): molecular transmission squared as function of range
    '''
    # generate profile
    if genboo:
        # reading scattering coefficient file
        ray_file = DIRCONFN(RAYLEIGHPROFILEDIR,
                            RAYLEIGHCDFDIR.format(RAYLEIGHCDLAMBDA))
        rayscatdat_nc = nc.Dataset(ray_file, 'r', format='NETNC4_CLASSIC')
        ncr_ra = rayscatdat_nc.variables['range'][:]
        betam_ra = rayscatdat_nc.variables[weather+'_ray'][:]

        # interpolating
        Delr = Delt * SPEEDOFLIGHT
        r_ra = Delr * np.arange(Nbin) + Delr/2
        betam_ra = np.interp(r_ra, ncr_ra, betam_ra)

        # computing back scatter
        betam_ra *= ((wavelength/RAYLEIGHCDLAMBDA)**(-4))

        # computing transmission
        sigmam_ra = _mollidrat * betam_ra              # scat cross sec
        Tm2_ra = np.exp(-2 * cumtrapz(sigmam_ra, r_ra, initial=0))

        # computing product
        betamprime_ra = betam_ra * Tm2_ra

        # writing to file
        np.savetxt(
            DIRCONFN(RAYLEIGHPROFILEDIR,
                     RAYLEIGHPROFILE.format(weather, wavelength, Delt, Nbin)),
            [betam_ra, Tm2_ra, betamprime_ra]
        )

        # returning
        return main(Delt, Nbin, False, weather, wavelength,
                    verbboo=False)


    # reading from exisiting file
    else:
        return np.loadtxt(
            DIRCONFN(RAYLEIGHPROFILEDIR,
                     RAYLEIGHPROFILE.format(weather, wavelength, Delt, Nbin))
        )


# generating
if __name__ == '__main__':
    Delt = 1e-7
    Nbin = 1200
    Delr = Delt * SPEEDOFLIGHT
    r_ra = Delr * np.arange(Nbin) + Delr/2
    betam_ra, T2m_ra = main(True, Delt, Nbin)
