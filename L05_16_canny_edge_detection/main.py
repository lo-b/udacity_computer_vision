"""
Canny edge detection demo.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Lees de foto in
image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"
image = cv2.imread(image_dir + 'brain_MR.jpg')

# verander kleur naar RGB (van BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# verander alle waardes naar een grijswaarde.
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()

# Implementatie canny edge:

# Voer canny edge uit met een nauw en een wijd verschil tussen
# de hoge en lage threshold
wide = cv2.Canny(gray, 30, 100)  # 70 verschil
tight = cv2.Canny(gray, 200, 240)  # 40 verchil

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
f.show()


