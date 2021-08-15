import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotation_image(src_img, angle, shape_out_img):

    width, height = src_img.shape[0], src_img.shape[1]
    # tranformation matrix
    R_mat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    result_img = np.zeros(shape_out_img, dtype="u1")

    for i in range(width):
        for j in range(height):

            _x = np.array([i, j, 1])
            # homogenrous vector
            new_xy = np.dot(R_mat, _x)

            if 0 <= new_xy[0] <= width and 0 <= new_xy[1] <= height:
                result_img[int(new_xy[0]), int(new_xy[1])] = src_img[i, j]

    return result_img


src_img = cv2.imread("/home/dat/data/lenna.png")
angle = 1
shape_out_img = src_img.shape

result_img = rotation_image(src_img, angle, shape_out_img)

cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
