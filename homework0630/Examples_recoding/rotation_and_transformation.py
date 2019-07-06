import cv2
import random
import numpy as np


image_location = 'C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg'

# rotation
img = cv2.imread(image_location)
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
print(M)

# set M[0][2] = M[1][2] =0
M[0][2] = 0
M[1][2] = 0
print(M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna 2', img_rotate2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# scale + rotation + translation = similarity transform
M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 30, 0.5)
img_rotate3 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna 3', img_rotate3)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

print(M)

###########################
# Affine transform
rows, cols, ch =img.shape
pts1 = np.float32([[0, 0], [cols -1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1 ],[cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affined lenna', dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#########################
# perspective transform


def random_warp(img):
    height, width, channels = img.shape

    # warp
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin -1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin -1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts11 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts22 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])

    m_warp1 = cv2.getPerspectiveTransform(pts11, pts22)
    img_warp2 = cv2.warpPerspective(img, m_warp1, (width, height))

    return m_warp1, img_warp2


M_warp, img_warp = random_warp(img)
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

