"""
Oefening met canny edge detection
"""
import cv2

from main import image_dir
import matplotlib.pyplot as plt

image = cv2.imread(image_dir + 'sunflower.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(image, cmap='gray')
plt.show()

# Thresholds definiëren:
# ratios van 1:2 tot 1:3 zouden de beste resultaten krijgen (source: course video)
lower = 200
upper = 240

edges = cv2.Canny(gray, lower, upper)

plt.imshow(edges, cmap='gray')
plt.show()
