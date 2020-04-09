import matplotlib.pyplot as plt

import numpy as np
import cv2

# Lees de foto in
image = cv2.imread('car_green_screen.jpg')
plt.title('Image with BGR colors')
# Print dimensies
print('Image dimensions:', image.shape)

# Transformeer foto van BGR naar RGB
image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(image_copy)
plt.title('Image with RGB colors')
plt.show()

# Definieër color threshold voor de kleur groen
lower_green = np.array([0, 70, 0])
upper_green = np.array([180, 255, 180])

# Bepaal de oppervlakte van de mask
mask = cv2.inRange(image_copy, lower_green, upper_green)

# %%
plt.imshow(mask, cmap='gray')
plt.show()

# Mask de foto om alleen de auto te laten zien
image_copy[mask != 0] = [0, 0, 0]

# %%
plt.imshow(image_copy)
plt.title("Image na masking")
plt.show()


background = cv2.imread('sky.jpg')
background_copy = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
print(background_copy.shape)
background_copy = background_copy[0:450, 0:660, 0:3]

background_copy[mask == 0] = [0, 0, 0]

# %%
plt.imshow(background_copy)
plt.show()

# Mijn zieke auto in de ruimte:
plt.imshow(image_copy + background_copy)
plt.show()

