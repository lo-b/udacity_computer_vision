import numpy as np
import matplotlib.pyplot as plt
import cv2

from L06_03_finding_corners.main import image_dir

image = cv2.imread(image_dir + 'monarch.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

print('Current image shape:', np.copy(image).shape)

# reshape foto naar een 2D array (momenteel is het een 3D array
# de 2d array bevat de pixels en de kleur wat drie waarden zijn
# voor r, g en b.
pixel_vals = image.reshape((-1, 3))

# een shape van (x, y) betekent een matrix  bij x bij y: x rijen
# en y kolommen
# De shape van pixel_vals is (x*y van de foto, 3)
# iedere pixel heeft dus drie kolommen waar de kolommen
# RGB voorstellen resp.
print('New shape: ', pixel_vals.shape)

# Maak van de integers floats
pixel_vals = np.float32(pixel_vals)

# Implementatie van k-means clustering


# definitie wanneer het algoritme moet stoppen
#  input:
#     - type: type voor WANNEER het moet stoppen
#       In het geval beneden stopen we wanneer de epsilon waarde
#       bereikt is (1.0 -- wat betekent 1 pixel verschil)
#     - maximaal aantal iteraties
#     - epsilon: gewenste accuratie
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3

# Input voor .kmeans()
# - de array
# - k: aantal clusters
# - labels: de labels voor de clusters; None betekent geen labels
# - criteria: criteria wanneer het algoritme moet stoppen
# - aantal pogingen
# - op welke manier de begin punten moeten worden bepaald
retval, labels, centers = cv2.kmeans(pixel_vals, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Terugconverteren naar 8-bit waarden (waarde tussen 0-255 wat
# we gebruiken om fotos op te slaan en weer te geven)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data in originele dimensie van de foto (x, y, 3)
segmented_image = segmented_data.reshape(image.shape)
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)
plt.show()

# Na het segmenteren van een foto kun je interessante dingen doen:
# zoals een cluster masken.
cluster = 0

masked_image = np.copy(image)

# maak de mask groen (glow in de dark butterfly maken)
masked_image[labels_reshape == cluster] = [0, 255, 0]

plt.imshow(masked_image)
plt.show()
