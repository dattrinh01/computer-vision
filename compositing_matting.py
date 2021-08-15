import cv2
import numpy as np

mark = cv2.imread("/home/dat/data/mask.png")
aim = cv2.imread("/home/dat/data/aim.png")

mark = cv2.resize(mark, (199, 199))
aim = cv2.resize(aim, (199, 199))


def compositing_and_matting(mark_img, aim_img, alpha):
    result = np.zeros((199, 199, 3), dtype="u1")
    for i in range(199):
        for j in range(199):
            temp = (1 - alpha) * mark_img[i, j] + alpha * aim_img[i, j]
            result[i, j] = temp
    return result


result = compositing_and_matting(mark, aim, 0.5)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
