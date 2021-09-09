import cv2
import numpy as np
import matplotlib.pyplot as plt


src_img = cv2.imread('lenna.png')


def convolution_process(src_img, kernel):
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
    

def convol(src_img, filter):
    result_image = np.zeros(src_img.shape, dtype = np.float32)
    result_image[:, :] = convolution_process(src_img[:, :], filter)
    return result_image


def sobel_edge_detection(src_img, filter):
    vertical_filter = convol(src_img, filter)
    horizontal_filter = convol(src_img, np.flip(filter.T, axis = 0))
    gradient_magnitude = np.sqrt(np.square(vertical_filter) + np.square(horizontal_filter))
    gradient_magnitude *= 255 / gradient_magnitude.max()
    return vertical_filter


gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
filter = np.float32([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
result_image = sobel_edge_detection(gray_img, filter)

plt.imshow(result_image)
plt.show()
