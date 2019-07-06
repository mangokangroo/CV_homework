import cv2
import random
import numpy as np

#image_location = 'C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg'
image_location = input("Please enter the location of the image to be processed: ")
# load picture
img = cv2.imread(image_location)

# image crop
img_crop = img[0:150, 0:200]
cv2.imshow('cropped image', img_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# image color shift
def color_shift(img_crop_f, r_scaling, g_scaling, b_scaling):
    B, G, R = cv2.split(img_crop_f)
    # change Red component
    r_rand = random.randint(-10, 10) * r_scaling
    print('The random shifting of RED is ' + str(r_rand))
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    # change Green component
    g_rand = random.randint(-10, 10) * g_scaling
    print('The random shifting of GREEN is ' + str(g_rand))
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
    # chang Blue component
    b_rand = random.randint(-10, 10) * b_scaling
    print('The random shifting of BLUE is ' + str(b_rand))
    if b_rand == 0:
        pass
    elif b_rand >0:
        lim = 255 - b_rand
        B[B > lim] =255
        B[B <= lim] =(b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    img_shift = cv2.merge((B, G, R))
    return img_shift

r_shift = int(input("Please input the scaling factor of RED color shift: "))   # the scaling of red color shift
g_shift = int(input("Please input the scaling factor of GREEN color shift: "))    # the scaling of red color shift
b_shift = int(input("Please input the scaling factor of BLUE color shift: "))  # the scaling of red color shift

img_color_shift = color_shift(img_crop, r_shift, g_shift, b_shift)
cv2.imshow('color shift image', img_color_shift)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# rotation
def img_rotate(deg):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1)
    img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotated

rotate_angle = float(input("Please enter the angle for rotation: "))

img_rot = img_rotate(rotate_angle)
cv2.imshow('rotated image', img_rot)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# perspective transformation
def random_warp(img_to_warp, random_margin):
    height, width, channels = img_to_warp.shape

    # warp
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
perspective_margin = int(input("Please input the magin value of perspective transformation "))
M_warp, img_warp = random_warp(img_rot, perspective_margin)
cv2.imshow('lenna_warp', img_warp)
print(M_warp)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()