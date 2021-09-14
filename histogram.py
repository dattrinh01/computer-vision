from typing import ChainMap
from numpy.lib.function_base import histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread("lenna.png")

def histogram_img(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.plot(histogram)
    plt.show()

def digital_negative(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(255 - gray_img, cmap = "gray")
    plt.show()


def constrast_stretching(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    width, height = gray_img.shape
    min = np.min(gray_img)
    max = np.max(gray_img)

    result_img = np.zeros(gray_img.shape, dtype = "uint8")
    for i in range(0, width):
        for j in range(0, height):
            result_img[i, j] = (255 / (max - min)) * (gray_img[i, j] - min)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(cv2.calcHist([gray_img], [0], None, [256], [0, 256]))
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(cv2.calcHist([result_img], [0], None, [256], [0, 256]))
    plt.show()
    
    cv2.imshow("Original image", gray_img)
    cv2.imshow("Contrast stretching image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

constrast_stretching(src_img)