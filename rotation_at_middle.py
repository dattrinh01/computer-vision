import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("lenna.png")


def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def get_grid(height, width):
    coords = np.indices((height, width)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1])))

def rotation_processing(img):
    height, width = img.shape[0], img.shape[1]
    coords = get_grid(height, width)
    x_ori, y_ori = coords[0], coords[1]
    R_mat = rotation_matrix(np.radians(90))
    T_mat = translation_matrix(height // 2, width // 2)
    
    A = T_mat @ R_mat @ np.linalg.inv(T_mat) @ coords

    x_rotation, y_rotation = A[0, :], A[1, :]

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

