import matplotlib.pyplot as plt
import numpy as np
import cv2

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_3_Types_of_Features_Image_Segmentation/images/"

image = cv2.imread(image_dir + 'waffle.jpg')

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

# Converteer de pixels naar grijswaarden
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# Het algoritme werkt op floats en niet op ints
# dus we converteren elke waarde naar een float
gray = np.float32(gray)

# Detecteer hoeken door gebruik te maken van corner harris
# algoritme. Het neemt de foto met float grijswaardes, de grote van een zijde
# van het gebied (2 ==> vierkant van 2 bij 2px), de shape van de
# sobel operators ( 3 ==> 3 bij 3 matrix) en een constante die
# bepaalt waarneer iets een hoek is -- een waarde van 0.04 werkt goed.
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

plt.imshow(dst, cmap='gray')
plt.show()

# Als output van corner harris krijgen we dezelfde foto
# waar hoeken fel wit worden gemaakt; om dit beter te zien
# dialten ( verwijden) we de foto: we laten het felle
# wit meer naar de voorgrond komen
dst = cv2.dilate(dst, None)

plt.imshow(dst, cmap='gray')
plt.show()

# Geef de hoeken aan in het origineel door een treshold de definiëren
# wanneer een hoek een 'sterke' hoek is.
treshold = 0.04 * dst.max()

# Maak een kopie om overheen te tekenen
corner_image = np.copy(image_copy)

# Teken een cirkel wanneer een hoek boven de treshold zit:
for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if dst[j, i] > treshold:
            # arguments: foto, middelpunt, radius, kleur en dikte van de lijn
            cv2.circle(corner_image, (i, j), 1, (0, 255, 0), 1)

plt.imshow(corner_image)
plt.show()
