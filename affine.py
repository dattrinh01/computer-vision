import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("lenna.png")


def rotation_matrix(angle):
    return np.array([
        [0.1, 0.8, 0.009],
        [0.25, 1, 0.01],
        [0, 0, 1]
    ])

def get_grid(height, width):
    coords = np.indices((height, width)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1])))

def rotation_processing(img):
    height, width = img.shape[0], img.shape[1]
    coords = get_grid(height, width)
    x_ori, y_ori = coords[0], coords[1]
    R_mat = rotation_matrix(np.radians(45))
    
    R_mat_result = R_mat @ coords

    x_rotation, y_rotation = R_mat_result[0, :], R_mat_result[1, :]

    indices = np.where((x_rotation >= 0) & (x_rotation < height) & (y_rotation >= 0) & (y_rotation < width))

    x_ori_map, y_ori_map = x_ori[indices].astype(np.int32), y_ori[indices].astype(np.int32)
    x_rotation_map, y_rotation_map = x_rotation[indices].astype(np.int32), y_rotation[indices].astype(np.int32)

    canvas = np.zeros_like(img)
    canvas[x_ori_map, y_ori_map] = img[x_rotation_map, y_rotation_map]

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image original")
    plt.subplot(1, 2, 2)
    plt.imshow(canvas)
    plt.title("Image after rotation")
    plt.show()

rotation_processing(img)

