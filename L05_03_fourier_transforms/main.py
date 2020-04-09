"""
Fourier Transform demo.

Een Fourier transform is op een ANDERE MANIER
dezelfde data laten zien.
"""

import matplotlib.pyplot as plt
import cv2

# Lees fotos in
from .fourier import ft_image

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"

image_stripes = cv2.imread((image_dir + 'stripes.jpg'))
image_solid = cv2.imread(image_dir + 'pink_solid.jpg')

# Verander kleurmodel
image_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_BGR2RGB)
image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)

# Laat de fotos zien
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(image_stripes)
ax2.imshow(image_solid)
f.show()

# Converteer RGB naar grijswaarden om inzicht te krijgen in
# de intensiteit

gray_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)
gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)

# Transformeer de normale grijswaarde tussen 0 en 255 naar
# een waarde tussen de 0 en 1.
norm_stripes = gray_stripes / 255.0
norm_solid = gray_solid / 255.0

# Bereken transforms
f_stripes = ft_image(norm_stripes)
f_solid = ft_image(norm_solid)

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.set_title('original image')
ax1.imshow(image_stripes)
ax2.set_title('frequency transform image')
ax2.imshow(f_stripes, cmap='gray')

ax3.set_title('original image')
ax3.imshow(image_solid)
ax4.set_title('frequency transform image')
ax4.imshow(f_solid, cmap='gray')

f.show()

# Fourier transform op een foto van vogels
image = cv2.imread(image_dir + 'birds.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

norm_image = gray / 255.0

# De witte vlek in de FT-visualisatie betekent dat grotendeels
# van de foto een lage frequentie heeft.
# Daarnaast zijn er twee richtingen voor deze frequenties:
#   - verticale randen (van de vogels) worden weergegeven
#     met een horizontale witte streep
#   - horizontale randen (van de stok waar de vogels op zitten)
#     worden weergegeven met de verticale witte streep.
f_image = ft_image(norm_image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(image)
ax2.imshow(f_image, cmap='gray')
f.show()
