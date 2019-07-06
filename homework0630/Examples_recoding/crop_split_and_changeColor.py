import cv2
import random

# load color image
img = cv2.imread('C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg')

# image crop
img_crop = img[0:150, 0:200]
cv2.imshow('img_crop',img_crop)
key=cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# image split
B, G, R = cv2.split(img)
cv2.imshow('img_B', B)
cv2.imshow('img_G', G)
cv2.imshow('img_R', R)
key=cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# change color
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)

    # change color B
    b_rand = random.randint(-50, 50)
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

    # change color G
    g_rand = random.randint(-50, 50)
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

    # change color R
    r_rand = random.randint(-50, 50)
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

    img_merge = cv2.merge((B, G, R))
    return img_merge

img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
key=cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()