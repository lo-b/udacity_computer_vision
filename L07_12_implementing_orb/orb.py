import matplotlib.pyplot as plt
import cv2
import copy
from L07_12_implementing_orb.visualize import *

plt.rcParams['figure.figsize'] = [14.0, 7.0]

# Zet de parameters van het ORB algoritme waar het meeste
# op default blijft staan en we alleen aangeven dat we 200 keypoints willen
# en een downsampling ratio van 4 hebben.
orb = cv2.ORB_create(200, 2.0)

# Vind de keypoints en bepaal de ORB discriptor
# None geeft aan dat we geen mask bepalen
keypoints, descriptor = orb.detectAndCompute(training_gray, None)

# Maak kopieën van de foto om de keypoints op te visualiseren
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

# Draw the keypoints without size or orientation on one copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color=(0, 255, 0))

# Draw the keypoints with size and orientation on the other copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with the keypoints without size or orientation
plt.subplot(121)
plt.title('Keypoints Without Size or Orientation')
plt.imshow(keyp_without_size)

# Display the image with the keypoints with size and orientation
plt.subplot(122)
plt.title('Keypoints With Size and Orientation')
plt.imshow(keyp_with_size)
plt.show()

# Print the number of keypoints detected
print("\nNumber of keypoints Detected: ", len(keypoints))
