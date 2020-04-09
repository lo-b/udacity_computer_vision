import numpy as np
import matplotlib.pyplot as plt
import cv2

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"

image = cv2.imread(image_dir + 'brain_MR.jpg')

image_copy = np.array(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

# Gaussian filter werkt alleen op een foto met grijswaarden.
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

averaging_filter = np.ones((9, 9)) * (1 / (9 * 9))

# Averaging filter om het verschil te zien
blur_using_averaging_filter = cv2.filter2D(gray, -1, averaging_filter)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.set_title('original gray')
ax1.imshow(gray, cmap='gray')

ax2.set_title('blurred image using gaussian blur')
ax2.imshow(gray_blur, cmap='gray')

ax3.set_title('blurred image using averaging filterblur')
ax3.imshow(blur_using_averaging_filter, cmap='gray')

f.show()

# Kijken hoe edges weer worden gegeven na gebruiken van
# high-pass filters (Sobel x & y)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Pas de Sobel filters toe op het origineel, foto blur'd door
# de Gaussian blur en de averaging filter.

filtered = cv2.filter2D(gray, -1, sobel_x)
filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_x)
filtered_averaging = cv2.filter2D(blur_using_averaging_filter, -1, sobel_x)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.set_title('original gray')
ax1.imshow(filtered, cmap='gray')
ax2.set_title('blurred image')
ax2.imshow(filtered_blurred, cmap='gray')
ax3.set_title('blurred image by averaging filter')
ax3.imshow(filtered_averaging, cmap='gray')

# In deze plot is goed te zien waarom de Gaussian vaak
# wordt gebruikt als een pre processing stap.
f.show()
