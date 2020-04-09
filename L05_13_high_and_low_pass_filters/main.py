import numpy as np
import matplotlib.pyplot as plt
import cv2

# Definieer wat filters:
from matplotlib.cm import cmap_d

gaussian_shape = (3, 3)
gaussian = (1 / (gaussian_shape[0] * gaussian_shape[1])) * np.ones(gaussian_shape)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# laplacian, edge filter
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

filters = [gaussian, sobel_x, sobel_y, laplacian]
filters_name = ['gaussian', 'sobel_x', 'sobel_y', 'laplacian']

f_filters = [np.fft.fft2(x) for x in filters]
fshift = [np.fft.fftshift(y) for y in f_filters]
frequency_tx = [np.log(np.abs(z) + 1) for z in fshift]

# Laat alle filters zien in een plot
for i in range(len(filters)):
    plt.subplot(2, 2, i + 1), plt.imshow(frequency_tx[i], cmap='gray')
    plt.title(filters_name[i]), plt.xticks([]), plt.yticks([])

plt.show()

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"

image = cv2.imread(image_dir + 'sunflower.jpg')

image_copy = np.array(image)
image_RGB = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
# image_blur = cv2.GaussianBlur(image_gray, (9, 9), 0)

filtered_image = cv2.filter2D(image_gray, -1, gaussian)
norm_image = filtered_image / 255.0


# Functie om fourier transform te berekenen (uit vorige notebook)
def ft_image(norm_image):
    """
    Functie die als input een genormaliseerde foto neemt
    en een frequentie spectrum transform als output geeft.
    """
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20 * np.log(np.abs(fshift))

    return frequency_tx


f_image_original = ft_image(image_gray / 255.0)
f_image = ft_image(norm_image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title('FT op originele grijsfoto')
ax1.imshow(f_image_original, cmap='gray')

ax2.set_title('FT op grijsfoto waar filter op is toegepast')
ax2.imshow(f_image, cmap='gray')

f.show()
