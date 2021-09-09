import cv2
import numpy as np
import math

src_img = cv2.imread("data_noise.png")

gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

def process(src_img):
    outputImg = src_img.copy()
    width, height = src_img.shape

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            temp = [
                src_img[i - 1, j], src_img[i - 1, j], src_img[i - 1, j + 1],
                src_img[i, j - 1], src_img[i, j], src_img[i, j + 1],
                src_img[i + 1, j - 1], src_img[i + 1, j], src_img[i + 1, j + 1]
            ]
            temp = sorted(temp)
            outputImg[i, j] = temp[4]
    return outputImg


result = process(gray)
cv2.imshow("Img", result)
cv2.waitKey(0)
cv2.destroyAllWindows()