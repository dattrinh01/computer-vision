import cv2
import numpy as np


def scale_img(src_img, scale_values, shape_out_img):
    width, height = src_img.shape[0], src_img.shape[1]
    S_mat = np.array([
        [scale_values[0], 0],
        [0, scale_values[1]]
    ])

    result_img = np.zeros(shape_out_img, dtype='u1')

    for i in range(width):
        for j in range(height):

            _x = np.array([i, j])
            new_xy = np.dot(S_mat, _x)

            if 0 < new_xy[0] < width and 0 < new_xy[1] < height:
                result_img[new_xy[0], new_xy[1]] = src_img[i, j]

    return result_img


src_img = cv2.imread("lenna.png")
shape_out_img = src_img.shape
scale_values = [2, 1]

result_img = scale_img(src_img, scale_values, shape_out_img)

cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
