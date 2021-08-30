import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

src_img = cv2.imread("lenna.png")
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

def gaussian_blur_1(kernel_size, sigma):
    x = np.linspace(-sigma, sigma, kernel_size + 1)
    x = np.diff(st.norm.cdf(x))
    kernel = np.outer(x,x)
    return kernel/kernel.sum()

def gaussian_blur_2(kernel_size, sigma):
    kernel = np.ones((kernel_size,kernel_size))
    kernel_size = kernel_size // 2
    for x in range(-kernel_size, kernel_size + 1):
        for y in range(-kernel_size, kernel_size + 1):
            part_1 = np.exp(-(x**2+y**2)/(2*sigma**2))
            part_2 = 2 * np.pi * sigma**2
            kernel[x+kernel_size, y+kernel_size] = part_1 / part_2
    return kernel

def conv2D(src_img, gau_mat, kernel_size):
    [row, col] = src_img.shape
    result = np.zeros(src_img.shape, dtype = "u1")
    for i in range(kernel_size, row):
        for j in range(kernel_size, col):
            temp = src_img[i, j] * gau_mat
            result[i,j] = temp.sum()
    return result



kernel1 = gaussian_blur_1(5,2)
kernel2 = gaussian_blur_2(5,1.1)
result_img_1 = cv2.filter2D(src_img,-1,kernel1)
result_img_2 = cv2.filter2D(src_img,-1,kernel2)


cv2.imshow("img", result_img_1)
cv2.imshow("img1",result_img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()