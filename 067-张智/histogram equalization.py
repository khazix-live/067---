import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("lena.png")
# 灰度化
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_his = cv.calcHist([gray_img], [0], None, [256], [0,256])
# 图像均衡化
equal_img = cv.equalizeHist(gray_img)
equal_his = cv.calcHist([equal_img], [0], None, [256], [0,256])

plt.figure()
plt.title("gray histogram")
plt.hist(gray_img.ravel(), 256)
plt.show()

plt.figure()
plt.title("equal histogram")
plt.hist(equal_img.ravel(), 256)
plt.show()

cv.imshow("Histogram Equalization", np.hstack([gray_img, equal_img]))
cv.waitKey(0)
