import os

from src import helpers
import matplotlib.pyplot as plt


def visualize_data(image_list):
    """
    Het is ongelofelijk belangrijk om data te visualiseren voordat
    je begint met het trainen van een model.

    Deze functie laat info over de lijst van fotos zien en laat
    een dag en nachtfoto zien
    """
    path, dirs, files = next(os.walk("../training/day"))
    file_count = len(files)
    path, dirs, files = next(os.walk("../training/night"))
    file_count += len(files)

    print('Er zijn', file_count, 'fotos voor training\n'
                                 '120 dag-fotos & 120 '
                                 'nacht-fotos.')

    path, dirs, files = next(os.walk("../test/day"))
    file_count = len(files)
    path, dirs, files = next(os.walk("../test/night"))
    file_count += len(files)

    print('Er zijn', file_count, 'fotos voor testen\n'
                                 '80 dag-fotos & 80 '
                                 'nacht-fotos.')

    # Laat de eerste foto zien met wat info
    image_index = 0
    selected_image = image_list[image_index][0]
    selected_label = image_list[image_index][1]

    # Laat de foto zien:
    # %%
    plt.imshow(selected_image)
    plt.show()
    print('Dimensies van de foto:', selected_image.shape)
    print('Label van de foto:', selected_label)

    # Zoek de eerste nachtfoto en laat deze ook zien
    for pair in image_list:
        if pair[1] == "night":
            plt.imshow(pair[0])
            plt.show()
            break


def visualize_averages(image_list):
    night_images_y = []  # Gemiddelde waardes
    day_images_y = []

    night_images_x = list(range(0, 120))

    day_images_x = list(range(0, 120))

    night_images = []
    day_images = []

    for t in image_list:
        image = t[0]
        avg = helpers.avg_brightness(image)

        if t[1] == 'night':
            night_images_y.append(avg)
            night_images.append(image)
        else:
            day_images_y.append(avg)
            day_images.append(image)

    plt.bar(day_images_x, day_images_y)
    plt.title('Gemiddelde tijdens dag')
    plt.show()

    plt.bar(night_images_x, night_images_y)
    plt.title('Gemiddelde tijdens de nacht')
    plt.show()
