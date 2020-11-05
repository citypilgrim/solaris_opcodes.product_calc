# imports
import numpy as np
import scipy.signal as sig


# main func
def main(
        work_tra, z_ra, r_rm, setz,
        lowpasspoly, lowpassfirstcut, lowpasssecondcut,
        savgolwindow, savgolpoly,
):
    '''
    performs noise filtering on working chunk

    Parameters
        work_tra (np.ndarray): array to be filtered in the range axis
        z_ra (np.ndarray): 1d corresponding array
        r_rm (np.ndarray): 1d mask corresponding array
        lowpasspoly (int): polynomial order for lowpass filter
        lowpassfirstcut (float): first lowpass freq cut off
        lowpasssecondcut (float): second lowpass freq cut off
        savgolwindow (int): window size for savgol filter, must be odd number
        savgolpoly (int): polynomial order for savgol filter
        padding (int): padding of '0's to add to the front of the return array
    Return
        ret_tra (np.ndarray): filtered work_tra
        z_ra (np.ndarray): corresponding range array
        r_rm (np.ndarray): corresponding mask array, all True
        setz (list): unchanged
    '''
    work_tra = work_tra[:, r_rm]
    z_ra = z_ra[r_rm]

    # low pass filter
    T = np.abs(z_ra[1] - z_ra[0])
    fs = 1/T          # sample rate, [km^-1]
    nyq = 0.5 * fs              # Nyquist Frequency
    lowpass1_tra = work_tra
    b, a = sig.butter(lowpasspoly, lowpassfirstcut/nyq, btype='low', analog=False)
    lowpass1_tra = sig.filtfilt(b, a, lowpass1_tra, axis=-1)

    # savgol filter
    savgol_tra = lowpass1_tra
    savgol_tra = sig.savgol_filter(savgol_tra, savgolwindow, savgolpoly, axis=-1)

    # low pass filter
    lowpass2_tra = savgol_tra
    b, a = sig.butter(lowpasspoly, lowpasssecondcut/nyq, btype='low', analog=False)
    lowpass2_tra = sig.filtfilt(b, a, lowpass2_tra, axis=-1)

    # handling invalid values
    ret_tra = lowpass2_tra
    # ret_tra[ret_tra < 0] = 0

    # changing setz
    setz[1] = z_ra.shape[0]  # number of bins has changed

    return ret_tra, z_ra, np.ones_like(z_ra, dtype=np.bool), setz


# testing
if __name__ == '__main__':
    main()
