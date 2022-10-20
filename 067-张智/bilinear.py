import cv2 as cv
import numpy as np
# 获得原图像数据
img = cv.imread('lena.png')
src_h, src_w, channel = img.shape

# goal_h= int(input("输入目标图片的高"))
# goal_w= int(input("输入目标图片的宽"))
# goal_img = np.zeros((goal_h, goal_w, channel), np.uint8)

# 设定目标图像数据
goal_img = np.zeros((600, 600, channel), np.uint8)
dst_h, dst_w, dst_channel = goal_img.shape
print(f"src_h = {src_h},scr_w = {src_w},dst_h = {dst_h}, dst_w = {dst_w}")

# 比例系数
factor_h, factor_w = src_h/dst_h, src_w/dst_w
for i in range(3):
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            src_x = (dst_x + 0.5) * factor_w-0.5
            src_y = (dst_y + 0.5) * factor_h - 0.5

            src_x0 = int(np.floor(src_x))    # 四舍五入并取整
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, src_h - 1)

            temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
            temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
            goal_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)


cv.imshow("before",img)
cv.imshow("after",goal_img)
cv.waitKey(0)
