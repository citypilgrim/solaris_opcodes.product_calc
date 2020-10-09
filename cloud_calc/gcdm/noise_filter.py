# imports
import scipy.signal as sig

# params


# main func
def main(
        input_a, inputz_a,
        lowpasspoly, lowpassfirstcut, lowpasssecondcut,
        savgolwindow, savgolpoly,
):
    '''
    performs filtering of noise on a 1d array

    Parameters
        input_a (np.ndarray): 1d array to be filtered
        inputz_a (np.ndarray): 1d corresponding array
        lowpasspoly (int): polynomial order for lowpass filter
        lowpassfirstcut (float): first lowpass freq cut off
        lowpasssecondcut (float): second lowpass freq cut off
        savgolwindow (int): window size for savgol filter, must be odd number
        savgolpoly (int): polynomial order for savgol filter
    '''
    # low pass filter
    fs = 1/lowpasspoly          # sample rate, [km^-1]
    nyq = 0.5 * fs              # Nyquist Frequency
    order = lowpasspoly
    lowpass1_ra = input_a
    b, a = sig.butter(order, lowpassfirstcut/nyq, btype='low', analog=False)
    lowpass1_ra = sig.filtfilt(b, a, lowpass1_ra)

    # savgol filter
    savgol_ra = lowpass1_ra
    savgol_ra = sig.savgol_filter(savgol_ra, savgolwindow, savgolpoly)

    # low pass filter
    order = lowpasspoly
    lowpass2_ra = input_a
    b, a = sig.butter(order, lowpasssecondcut/nyq, btype='low', analog=False)
    lowpass2_ra = sig.filtfilt(b, a, lowpass2_ra)

    return lowpass2_ra


# testing
if __name__ == '__main__':
    main()
