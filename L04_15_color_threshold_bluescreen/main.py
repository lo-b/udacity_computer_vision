import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read image
image = cv2.imread("pizza_bluescreen.jpg")

# Printing information about image
print('This image is:', type(image),
      ' with dimensions:', image.shape)

# Change color to RGB (from BGR)
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# Display image

# %%
plt.imshow(image_copy)
plt.show()

# Define the color threshold
# defining lower && upperbound thresholds

# Isolating blue
lower_blue = np.array([0, 0, 220])
upper_blue = np.array([50, 70, 255])

# Het maken van een mask
# Dit is het isoleren van een gebied in een foto.
#
# Als de waarde van een pixel binnen `upper` en
# `lower` ligt dan laat de mask deze zien (wit [255] in de plot)
# en anders niet (zwart [0] in de foto).
mask = cv2.inRange(image_copy, lower_blue, upper_blue)


# %%
plt.imshow(mask, cmap='gray')
plt.show()


masked_image = np.copy(image_copy)

# Where the mask is not black, set it to black
masked_image[mask != 0] = [0, 0, 0]

# %%
plt.imshow(masked_image)
plt.show()

# Vervang de achtergrond
# Laad een achtergrondfoto in
new_background = cv2.imread('space_background.jpg')

# Transformeer van BGR naar RGB
new_background_copy = cv2.cvtColor(new_background, cv2.COLOR_BGR2RGB)

# Bijsnijden
new_background_cropped = new_background_copy[0:514, 0:816]

# Waar de mask zwart is willen we geen achtergrond hebben
# dus maken we dit zwart in de achtergrond.
new_background_cropped[mask == 0] = [0, 0, 0]

# Voeg de foto en achtergrond samen:
complete_image = new_background_cropped + masked_image

# %%
plt.imshow(complete_image)
plt.show()
