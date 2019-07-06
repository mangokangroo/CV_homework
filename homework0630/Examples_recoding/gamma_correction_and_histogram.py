import cv2
import numpy as np
from matplotlib import pyplot as plt


# gamma correction
img_dark = cv2.imread('C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg')
cv2.imshow('img_dark',img_dark)
key = cv2.waitKey()
if key ==27:
    cv2.destroyAllWindows()

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0)**inv_gamma)*255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)
img_brighter = adjust_gamma(img_dark, 2)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# histogram
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0] * 0.5), int(img_brighter.shape[1] * 0.5)))
plt.hist(img_brighter.flatten(), 256, [0,256], color = 'r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist((img_yuv[:,:,0]))
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized picture', img_output)
key = cv2.waitKey()
if key == 27:
    exit()
