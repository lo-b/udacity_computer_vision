import cv2
import matplotlib.pyplot as plt

image = cv2.imread('bee.jpg')

training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)


plt.rcParams['figure.figsize'] = [20, 10]

# Laat de training foto zien
plt.subplot(121)
plt.title('Original Training Image')
plt.imshow(training_image)
plt.subplot(122)
plt.title('Gray Scale Training Image')
plt.imshow(training_gray, cmap = 'gray')
plt.show()

