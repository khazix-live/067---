import cv2 as cv
import numpy as np
import random


# 图片，比例系数
def sp_noise(img,factor):
    noise_img = img
    noise_num = int(factor*noise_img.shape[0]*noise_img.shape[1])
    for i in range(noise_num):
        x = random.randint(0, noise_img.shape[0] - 1)
        y = random.randint(0, noise_img.shape[1] - 1)
        if random.random() <= 0.5:
            noise_img[x, y] = 0
        else:
            noise_img[x, y] = 255
    return noise_img

img = cv.imread('lenna.png')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
noise_img = sp_noise(img2, 0.3)
cv.imshow('原图', img)
cv.imshow('椒盐噪声', noise_img)
cv.waitKey(0)
