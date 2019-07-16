import cv2
import sys
import numpy as np


def medianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string

    # padding
    kernel_height = int(kernel[0])
    kernel_width = int(kernel[1])
    num_pad = int(kernel[0]/2)
    num_kernel_half_height = int(kernel[1]/2)
    pad_matrix = np.zeros((height + 2 * num_pad, width + 2 * num_pad))
    for i in np.arange(num_pad, height + num_pad):
        for j in np.arange(num_pad, width + num_pad):
            a = img[i - num_pad, j - num_pad]
            pad_matrix[i, j] = a
    if padding_way == 'ZEROS':
        pass
    # REPLICA PADDING
    elif padding_way == 'REPLICA':
        # four corners
        left_up = img[0, 0]
        left_down = img[height - 1, 0]
        right_up = img[0, width - 1]
        right_down = img[height - 1, width - 1]

        pad_matrix[0:num_pad - 1, 0:num_pad - 1] = left_up
        pad_matrix[0:num_pad, width + num_pad:] = right_up
        pad_matrix[height + num_pad:, 0: num_pad] = left_down
        pad_matrix[height + num_pad:, width + num_pad:] = right_down
        # rows padding
        for i in np.arange(num_pad, height + num_pad):
            lf = img[i - num_pad, 0]
            pad_matrix[i, 0:num_pad] = lf
            rt = img[i - num_pad, width - 1]
            pad_matrix[i, width + num_pad:] = rt
        # columns padding
        for j in np.arange(num_pad, width + num_pad):
            up = img[0, j - num_pad]
            pad_matrix[0:num_pad, j] = up
            down = img[height - 1, j - num_pad]
            pad_matrix[height + num_pad:, j] = down

    else:
        print('Padding way should be ZEROS or REPLICA. Please check the input of padding way')
        sys.exit()

    # print(pad_matrix)
    padded_output = pad_matrix.copy()
    # find median
    for i in np.arange(num_pad, height + num_pad):
        for j in np.arange(num_pad, width + num_pad):
            neighbors = []
            for k in np.arange(-num_pad, num_pad+1):
                for l in np.arange(-num_kernel_half_height, num_kernel_half_height+1):
                    b = pad_matrix[i+k, j+l]
                    neighbors.append(b)
            neighbors.sort()
            median = neighbors[int((kernel_height * kernel_width - 1) / 2)]
            padded_output[i, j] = median

    crop_padded_median = padded_output[num_pad: height + num_pad, num_pad: width + num_pad]
    crop_padded_median = crop_padded_median.astype(img.dtype)

    return crop_padded_median


# Import image
img_ori = cv2.imread('C:/Users/lxh/Desktop/AI/classes/class2_0706/lenna.jpg')
cv2.imshow('original lenna', img_ori)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the height and width of the image
height = img_ori.shape[0]
width = img_ori.shape[1]

# Get the kernel size and padding way from keyboard
blur_kernel_size = input("Please input the size of sliding window for median blurring: \n\
    (Note: 1. Please input two integer numbers separated by a SINGLE SPACE \n\
    2. Both should be ODD numbers, like 5 3 \n\
    3. The first should be NO LESS THAN the second) \n")
blur_kernel_size = blur_kernel_size.split(" ")
blur_kernel_size = [int(blur_kernel_size[i]) for i in range(len(blur_kernel_size))]
padding_approach = input("Please select the approach for padding, ZEROS or REPLICA: \n")

# Split the image into 3 channels, do medianBlur separately
B, G, R = cv2.split(img_ori)
B_blurred = medianBlur(B, blur_kernel_size, padding_approach)
G_blurred = medianBlur(G, blur_kernel_size, padding_approach)
R_blurred = medianBlur(R, blur_kernel_size, padding_approach)

# Show the image after median blur
img_blurred = cv2.merge((B_blurred, G_blurred, R_blurred))
cv2.imshow('image after median blur', img_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
