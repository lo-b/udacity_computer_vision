"""
Demo: werken met kernels
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/" \
            "1_2_Convolutional_Filters_Edge_Detection/images/"
image = mpimg.imread(image_dir + 'brain_MR.jpg')

plt.imshow(image)
plt.show()

blurred_img = cv2.blur(image, (10, 10))

gray = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale foto")
plt.show()

# 3x3 array for horizontal edge detection
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Create and apply a Sobel x operator
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


plt.imshow(blurred_img)
plt.title("Blurred img")
plt.show()

filtered_image = cv2.filter2D(blurred_img, -1, sobel_x)

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
second_filtered_img = cv2.filter2D(filtered_image, -1, sobel_y)

plt.imshow(second_filtered_img, cmap='gray')
plt.show()
