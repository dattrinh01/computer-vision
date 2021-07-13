import cv2
import numpy as np
import matplotlib.pyplot as plt


def translation_image(src_img, shift_distance, shape_out_img):
    width, height = src_img.shape[0], src_img.shape[1]
    T_mat = np.array([
        [1, 0, shift_distance[0]],
        [0, 1, shift_distance[1]],
    ])

    result_img = np.zeros(shape_out_img, dtype='u1')

    for i in range(width):
        for j in range(height):

            _x = np.array([i, j, 1])
            new_xy = np.dot(T_mat, _x)

            if 0 < new_xy[0] < width and 0 < new_xy[1] < height:
                result_img[new_xy[0], new_xy[1]] = src_img[i, j]
    return result_img


src_img = cv2.imread("lenna.png")

shift_distance = (10, 100)
shape_out_img = src_img.shape
result_img = translation_image(src_img, shift_distance, shape_out_img)

cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
