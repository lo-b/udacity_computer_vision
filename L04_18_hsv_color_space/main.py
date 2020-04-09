import numpy as np
import matplotlib.pyplot as plt
import cv2

# Lees foto in
image = cv2.imread('water_balloons.jpg')

# Converteer BGR naar RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(image)
plt.show()

# Plot RGB kanalen
r = image[:, :, 0]  # Neemt alle waardes voor de eerste en tweede kolom van het eerste kanaal op index 0.
g = image[:, :, 1]
b = image[:, :, 2]

# Plot de kanalen in een rij van 3 subplot.
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.set_title('Red')
ax1.imshow(r, cmap='gray')

ax2.set_title('Green')
ax2.imshow(g, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(b, cmap='gray')

f.show()  # Laat het hele figuur zien.

# Vervolgens hetzelde met HSV kleurmodel, let op van RGB naar HSV:

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.set_title('Hue')
ax1.imshow(h, cmap='gray')


ax2.set_title('Saturation')
ax2.imshow(s, cmap='gray')

ax3.set_title('Value')
ax3.imshow(v, cmap='gray')

f.show()

# Definiëer thresholds
# # Hue mask
lower_hue = np.array([160, 0, 0])
upper_hue = np.array([180, 255, 255])  # Hue is de hoek in de HSV cilinder en is dus een waarde tussen 0 en 180.

# Pink mask
lower_pink = np.array([180, 0, 100])
upper_pink = np.array([255, 255, 230])

# Maak masks
# Beginnend bij RGB:
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# Mask de fotos (selecteer de roze ballonen)
# Maak eerst een kopie van de foto en pas daarna de mask toe
mask_image = np.copy(image)
mask_image[mask_rgb == 0] = [0, 0, 0]

# Hue mask
mask_hue = cv2.inRange(hsv, lower_hue, upper_hue)

hsv_mask_image = np.copy(image)
hsv_mask_image[mask_hue == 0] = [0, 0, 0]

# %%
plt.imshow(mask_image)
plt.title('Masked image met RGB')
plt.show()
plt.imshow(hsv_mask_image)
plt.title('Masked image met HSV')
plt.show()
