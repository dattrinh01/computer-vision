import cv2
import numpy as np


def rotation_middle(src_img, shift_distance, angle, shape_out_img):
    width, height = src_img.shape[0], src_img.shape[1]

    T_mat = np.array([
        [1, 0, shift_distance[0]],
        [0, 1, shift_distance[1]],
        [0, 0, 1]
    ])

    R_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    transform_mat = T_mat @ R_mat @ np.linalg.inv(T_mat)

    result_img = np.zeros(shape_out_img, dtype='u1')

    for i in range(width):
        for j in range(height):

            _x = np.array([i, j, 1])
            new_xy = np.dot(transform_mat, _x)

            if 0 <= new_xy[0] < width and 0 <= new_xy[1] < height:
                result_img[new_xy[0].astype(np.int32), new_xy[1].astype(
                    np.int32)] = src_img[i, j]

    return result_img


src_img = cv2.imread("lenna.png")
shape_out_img = src_img.shape
shift_distance = (src_img.shape[0] // 2, src_img.shape[1] // 2)
angle = 45

result_img = rotation_middle(
    src_img, shift_distance, np.radians(angle), shape_out_img)

cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
