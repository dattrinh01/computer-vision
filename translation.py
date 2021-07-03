import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("lenna.png")

def get_grid(x, y):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1])))

def translation_image(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0 , 1]
    ])

def translation_processing(img):

    height, width = img.shape[:2]
    T_mat = translation_image(100, 100)
    coords = get_grid(height, width)
    x_original, y_original = coords[0], coords[1]

    T_mat_result = (T_mat @ coords).astype(np.int32)
    x_translation, y_translation = T_mat_result[0, :], T_mat_result[1, :]
    indices = np.where((x_translation >= 0) & (x_translation < height) & (y_translation >=0) & (y_translation < width))

    x_original_map, y_original_map = x_original[indices].astype(np.int32), y_original[indices].astype(np.int32)
    x_translation_map, y_translation_map = x_translation[indices].astype(np.int32), y_translation[indices].astype(np.int32)

    canvas = np.zeros_like(img)
    canvas[x_original_map, y_original_map] = img[x_translation_map, y_translation_map]

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image original")
    plt.subplot(1, 2, 2)
    plt.imshow(canvas)
    plt.title("Image after translation")
    plt.show()

translation_processing(img)