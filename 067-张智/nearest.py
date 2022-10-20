import cv2 as cv
import numpy as np
# 读取图片
img = cv.imread('lena.png')
# 获得图片的数据  高宽，通道数
height, width,channels = img.shape
# 预设放大图片尺寸
goal_img = np.zeros((400, 400, channels), np.uint8)
gh = 400/height
gw = 400/width

# 坐标映射
for i in range(400):
    for j in range(400):
        h = int(i/gh + 0.5)
        w = int(j/gw + 0.5)
        goal_img[i,j] = img[h,w]
cv.imshow('img',img)
cv.imshow('after',goal_img)
cv.waitKey(0)