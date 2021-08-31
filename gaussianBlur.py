import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

src_img = cv2.imread("lenna.png")

def gaussian(filter_size, sigma):
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    for x in range(-m, m+1):
        for y in range(-m, m+1):
            part_1 = 2 * np.pi * sigma**2
            part_2 = np.exp(-(x**2 + y**2)/2*sigma**2)
            gaussian_filter[x+m, y+m] = part_2 / part_1
    return gaussian_filter

def convolution(src_img, kernel):

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if (len(src_img.shape) == 3):
        pad_width = ((kernel_h//2, kernel_h//2) , (kernel_w//2, kernel_w//2) , (0,0))
        pad_image = np.pad(src_img, pad_width = pad_width, mode = "constant", constant_values = 0).astype(np.float32)
    elif (len(src_img.shape) == 2):
        pad_width = ((kernel_h//2, kernel_h//2), (kernel_w//2, kernel_w//2))
        pad_image = np.pad(src_img, pad_width = pad_width, mode = "constant", constant_values = 0).astype(np.float32)
    
    h = kernel_h // 2
    w = kernel_w // 2
    image_conv = np.zeros(pad_image.shape)

    for i in range(h, pad_image.shape[0] - h):
        for j in range(w, pad_image.shape[1] - w):
            x = pad_image[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i,j] = x.sum()
    
    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]
    
    return image_conv[h:h_end, w:w_end]

def process(src_image):
    gaussian_filter = gaussian(5,1)
    result_image = np.zeros(src_img.shape, dtype = "u1")
    for c in range(3):
        result_image[:, :, c] = convolution(src_img[:, :, c], gaussian_filter)
    return result_image
        

cv2.imshow("img", process(src_img))
cv2.waitKey(0)
cv2.destroyAllWindows()