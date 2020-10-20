# imports
import numpy as np
import scipy.signal as sig


# main func
def main(
        input_ra, inputz_ra, inputmask_ra,
        lowpasspoly, lowpassfirstcut, lowpasssecondcut,
        savgolwindow, savgolpoly,
        padding=0,
):
    '''
    performs filtering of noise on a 1d array

    Future
        - can parrellise the filtering parts, the same way did did gradient
          computation

    Parameters
        input_ra (np.ndarray): 1d array to be filtered
        inputz_ra (np.ndarray): 1d corresponding array
        inputmask_ra (np.ndarray): 1d mask corresponding array
        lowpasspoly (int): polynomial order for lowpass filter
        lowpassfirstcut (float): first lowpass freq cut off
        lowpasssecondcut (float): second lowpass freq cut off
        savgolwindow (int): window size for savgol filter, must be odd number
        savgolpoly (int): polynomial order for savgol filter
        padding (int): padding of '0's to add to the front of the return array
    Return
        ret_raa (np.ndarray): (4, masked range), array containing 2 arrays,
                              first is original, second is filtered array,
                              third is masked range array, fourth padding mask
    '''
    input_ra = input_ra[inputmask_ra]
    inputz_ra = inputz_ra[inputmask_ra]

    # low pass filter
    T = inputz_ra[1] - inputz_ra[0]
    fs = 1/T          # sample rate, [km^-1]
    nyq = 0.5 * fs              # Nyquist Frequency
    lowpass1_ra = input_ra
    b, a = sig.butter(lowpasspoly, lowpassfirstcut/nyq, btype='low', analog=False)
    lowpass1_ra = sig.filtfilt(b, a, lowpass1_ra)

    # savgol filter
    savgol_ra = lowpass1_ra
    savgol_ra = sig.savgol_filter(savgol_ra, savgolwindow, savgolpoly)

    # low pass filter
    lowpass2_ra = savgol_ra
    b, a = sig.butter(lowpasspoly, lowpasssecondcut/nyq, btype='low', analog=False)
    lowpass2_ra = sig.filtfilt(b, a, lowpass2_ra)

    # handling invalid values
    ret_ra = lowpass2_ra
    ret_ra[ret_ra < 0] = 0

    # adding padding
    pad = np.zeros(int(padding))
    ret_raa = np.array([
        np.append(pad, input_ra),
        np.append(pad, ret_ra),
        np.append(pad, inputz_ra),
        np.append(pad, np.ones(len(ret_ra)-int(padding)))
    ])

    return ret_raa


# testing
if __name__ == '__main__':
    main()
