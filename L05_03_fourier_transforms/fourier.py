import numpy as np


def ft_image(norm_image):
    """
    Functie die als input een genormaliseerde foto neemt
    en een frequentie spectrum transform als output geeft.
    """
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20 * np.log(np.abs(fshift))

    return frequency_tx
