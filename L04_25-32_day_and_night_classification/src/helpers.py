# Helper functions

import os
import glob  # library for loading images from a directory
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
import numpy as np


def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["day", "night"]

    # Iterate through each color folder
    for im_type in image_types:

        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            # Read in the image
            im = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list


## Standardize the input images
# Resize each image to the desired input size: 600x1100px (hxw).

## Standardize the output
# With each loaded image, we also specify the expected output.
# For this, we use binary numerical values 0/1 = night/day.


def standardize_input(image):
    """
    Preprocessing stap waar alle fotos 1100px breed en 600px hoog worden gemaakt.
    :param image:
    :return:
    """
    standard_im = cv2.resize(image, (1100, 600))
    return standard_im


def encode(label):
    """
    Encode de labels van een foto.
    :param label: het label van de foto
    :return: 0 als de foto genomen is in de nacht, anders 1
    """
    return 1 if label == 'day' else 0


def standardize(image_list):
    """
    Standaardiseert een lijst van fotos
    :param image_list: een lijst aan fotos
    :return: een lijst aan fotos die zijn gecropped en een integer numerieke label hebben.
    """
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # Create a numerical label
        binary_label = encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, binary_label))

    return standard_list


def avg_brightness(rgb_image):
    # Converteer foto naar HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Tel all value waardes bij elkaar op (laatste -- derde -- kanaal, dus index 2)
    brightness_sum = np.sum(hsv_image[:, :, 2])

    # Bereken de oppervlakte van de foto
    area = rgb_image.shape[0] * rgb_image.shape[1]

    return brightness_sum / area


def estimate_label(rgb_image):
    """
    Classificeert een rgb foto. Samen met de `avg_brightness` functiie is dit het algoritme wat bepaald
    of een foto een dag / nachtfoto is.
    :param rgb_image: de foto
    :return: een integer 0 als de foto een nachtfoto is; anders 1.
    """
    avg = avg_brightness(rgb_image)
    threshold = 100

    return 0 if avg <= threshold else 1


def visualize_hsv(standardized_list):
    # Eerst een visualisatie van een HSV foto waar we de h, s & v kanalen
    # plotten.
    test_img = standardized_list[190][0]
    test_label = standardized_list[190][1]

    # Converteren naar HSV
    test_hsv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)

    # Print het label
    print('HSV image label:', test_label)

    # Scheiden van channels
    h = test_hsv_img[:, :, 0]
    s = test_hsv_img[:, :, 1]
    v = test_hsv_img[:, :, 2]

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

    ax1.set_title('Standardized image')
    ax1.imshow(test_img)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')

    f.show()
