import numpy as np
import matplotlib.pyplot as plt
import cv2

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"

image = cv2.imread(image_dir + '/multi_faces.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()
plt.show()

# Haar cascade is een algoritme om gezichten te detecteren
# de parameters en architectuur worden gedefinieert in een xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# HOE gezichten gedetecteerd wordt, word door de methode
# `detectMultiScale` gedaan met als parameters: de foto, scaleFactor en minNeighbors.
faces = face_cascade.detectMultiScale(gray, 4, 6)

image_with_detections = np.copy(image)

# De output van de classifier is een array met waarden
# nodig om een rechthoek te tekenen (een punt, een hoogte en een breedte)
# de rechthoeken zijn altijd vierkanten!
for (x, y, w, h) in faces:
    cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 5)

plt.imshow(image_with_detections)
plt.show()