import numpy as np 
import matplotlib.pyplot as plt

img = plt.imread("lenna.png")

def scale_matrix(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

def get_grid(x, y):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1])))

def scale_processing(img):
    height, width = img.shape[0], img.shape[1]
    S_mat = scale_matrix(2)
    coords = get_grid(height, width)
    x_ori, y_ori = coords[0], coords[1]

    S_mat_result = S_mat @ coords

    x_scale, y_scale = S_mat_result[0, :], S_mat_result[1, :]

    indices = np.where((x_scale >= 0) & (x_scale < height) & (y_scale >= 0) & (y_scale < width))

    x_ori_map, y_ori_map = x_ori[indices].astype(np.int32), y_ori[indices].astype(np.int32)
    x_scale_map, y_scale_map = x_scale[indices].astype(np.int32), y_scale[indices].astype(np.int32)

    canvas = np.zeros_like(img)
    canvas[x_scale_map, y_scale_map] = img[x_ori_map, y_ori_map]

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image original")
    plt.subplot(1, 2, 2)
    plt.imshow(canvas)
    plt.title("Image after scale")
    plt.show()

scale_processing(img)