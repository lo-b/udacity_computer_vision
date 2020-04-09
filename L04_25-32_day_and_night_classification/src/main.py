import random
from src.helpers import *
from src.analyser import get_misclassified_images
from src.visualizer import visualize_data, visualize_averages

# VISUALISATIE
IMAGE_LIST = load_dataset('training')
visualize_data(IMAGE_LIST)

# PREPROCESSING
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# Visualisatie van foto converted van RGB naar HSV
visualize_hsv(STANDARDIZED_LIST)

# FEATURE EXTRACTION

# We gebruiken één feature voor de classifier: de helderheid (gemiddelde helderheid);
# eerst visualiseren we de helderheid van dag- en nachtfotos.
visualize_averages(IMAGE_LIST)

image_num = 190
test_img = STANDARDIZED_LIST[image_num][0]

avg = avg_brightness(test_img)

print('Avg brightness:', avg)
plt.imshow(test_img)
plt.show()


# ANALYSEER DE ACCURATIE VAN DE CLASSIFIER
TEST_IMAGE_LIST = load_dataset('test')

# Gebruik de eerder gemaakt standardize methode
STANDARDIZES_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Zet de test data op willekeurige volgorde
random.shuffle(STANDARDIZES_TEST_LIST)

MISCLASSIFIED = get_misclassified_images(STANDARDIZES_TEST_LIST)

total = len(STANDARDIZES_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct / total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))