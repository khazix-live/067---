import cv2 as cv
import numpy as np
import random

# 图片，上下限，比例系数
def Gaussian_noise(img,min,max,factor):
    noise_img = img
    noise_num = int(factor * noise_img.shape[0] * noise_img.shape[1])
    for i in range(noise_num):
        x = random.randint(0, noise_img.shape[0] - 1)
        y = random.randint(0, noise_img.shape[1] - 1)
        noise_img[x, y] = noise_img[x, y] + random.gauss(min, max)
        if noise_img[x, y] < 0:
            noise_img[x, y] = 0
        elif noise_img[x, y] > 255:
            noise_img[x, y] = 255
    return noise_img

img = cv.imread('lenna.png', 0)
cv.imshow("source", img)
gas_img = Gaussian_noise(img, 2, 4, 0.8)
img = cv.imread('lenna.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray img', gray_img)
cv.imshow('gas img', gas_img)
cv.waitKey(0)