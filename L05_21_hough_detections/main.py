"""
Demo Hough detection.

Soms is het nodig om kleinerre randen samen te voegen om goede
randdetectie te hebben; Hough detectie is een manier om randen,
van een image waar randen via high-pass filters al
weergegeven zijn, te verbinden.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

image_dir = "C:/Users/bram_/Desktop/CVND_Exercises/1_2_Convolutional_Filters_Edge_Detection/images/"

image = cv2.imread(image_dir + 'phone.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# Maak grijswaarde
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# threshold voor Canny
low_threshold = 50
high_threshold = 100

edges = cv2.Canny(gray, low_threshold, high_threshold)

plt.imshow(gray, cmap='gray')
plt.title("Foto met grijswaarden")
plt.show()

plt.imshow(edges, cmap='gray')
plt.title('Edges gevonden met Canny')
plt.show()

# Definieer variabele voor Hough transformatie
rho = 1  # afstand van oorsprong tot de lijn
theta = np.pi / 180
threshold = 60  # minimaal aantal snijpunten
min_line_length = 100
max_line_gap = 5

line_image = np.copy(image)

# Hough lines geeft een array met lijnen terug; iedere lijn
# bestaat uit twee punten x1, y1 en x2, y2
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

plt.imshow(line_image)
plt.show()
