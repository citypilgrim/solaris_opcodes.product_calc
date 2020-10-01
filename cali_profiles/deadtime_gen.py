# imports
import numpy as np

from ...global_imports.solaris_opcodes import *


# main func
@verbose
@announcer
def main(Ddir):
    '''
    Parameters
        Ddir (str): dir of .txt file containg deadtime coeff

    Return
        Dcoeff_a (np.array): array of deadtime fit coefficients
        D_f (func): takes in counts of any shape and out puts corr factor
    '''
    # generating Dtime coefficients
    with open(Ddir, 'r') as D_file:
        Dstr = D_file.read()
    Dstr = Dstr.replace('y = ', '').replace('- ', '-').replace('+ ', '')
    Dcoeff_l = Dstr.split(' ')[::-1]
    Dcoeff_a = np.array(list(map(
        lambda x: float(x[:x.find('x')]) if x.find('x') != -1 else\
        float(x), Dcoeff_l
    )))

    def D_f(n_ara):
        '''
        input of n_ara should be MHz, corrected to kHz in the function
        '''
        corr_ara = np.sum([
            Dcoeff * ((n_ara*1e3)**i) for i, Dcoeff in enumerate(Dcoeff_a)
        ], axis=0)

        return corr_ara

    return Dcoeff_a, D_f


# writing to profile to file
if __name__ == '__main__':
    from glob import glob
    import os.path as osp
    import matplotlib.pyplot as plt
    from ...global_imports.solaris_opcodes import *

    smmpl_boo = True
    if smmpl_boo:
        lidarname, mpl_d = 'smmpl_E2', '/home/tianli/SOLAR_EMA_project/data/smmpl_E2/calibration/SPCM37060deadtime.txt'
    else:
        lidarname, mpl_d = 'mpl_S2S', '/home/tianli/SOLAR_EMA_project/data/mpl_S2S/calibration/SPCM26086deadtime.txt'

    D_dirl = FINDFILESFN(DEADTIMEPROFILE, CALIPROFILESDIR)
    D_dirl.sort(key=osp.getmtime)
    D_dir = D_dirl[-1]
    Dcoeff_a, D_f = main(D_dir)

    Dsnstr = DIRPARSEFN(D_dir, fieldsli=DTSNFIELD)
    Dcoeff_fn = DEADTIMEPROFILE.format(Dsnstr, lidarname)
    np.savetxt(
        DIRCONFN(CALIPROFILESDIR, Dcoeff_fn),
        Dcoeff_a, fmt='%{}.{}e'.format(1, CALIWRITESIGFIG-1)
    )
