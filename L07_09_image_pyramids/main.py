import numpy as np
import matplotlib.pyplot as plt
import cv2

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_4_Feature_Vectors/images/"

image = cv2.imread(image_dir + 'rainbow_flag.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# Maak de image pyramid door gebruikt te maken van de downsampling
# methode .pyrDown()
level_1 = cv2.pyrDown(image)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))

ax1.set_title('original')
ax1.imshow(image)

ax2.imshow(level_1)
ax2.set_xlim([0, image.shape[1]])
ax2.set_ylim([image.shape[0], 0])

ax3.imshow(level_2)
ax3.set_xlim([0, image.shape[1]])
ax3.set_ylim([image.shape[0], 0])

ax4.imshow(level_3)
ax4.set_xlim([0, image.shape[1]])
ax4.set_ylim([image.shape[0], 0])

f.show()
