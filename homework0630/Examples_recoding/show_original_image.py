import cv2


# show grey, color image and their corresponding matrix
img_grey = cv2.imread('C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg',0)
cv2.imshow('lenna',img_grey)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# to show the grey level matrix of img_grey
print(img_grey)

# to show data type of the grey level matrix of img_grey
print(img_grey.dtype)

# to show the size of matrix of img_grey, height by width
print(img_grey.shape)

# load color image
img = cv2.imread('C:/Users/lxh/Desktop/AI/classes/class1_0630/lenna_304x306.jpg')
cv2.imshow('lenna_color',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
# to show color image and channels
print(img)
print(img.shape)

