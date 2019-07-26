import cv2
import numpy as np

img_ori = cv2.imread('Images/lenna.jpg')
grayImage = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
imageArray = np.asarray(grayImage)


def img_padding(img, kernel_size):
    height = len(img)
    width = len(img[0])
    num_pad = int(kernel_size / 2)
    padded_img = np.zeros((height + 2 * num_pad, width + 2 * num_pad))
    # pad the image matrix to the zero matrix
    for i in np.arange(num_pad, height + num_pad):
        for j in np.arange(num_pad, width + num_pad):
            a = img[i - num_pad][j - num_pad]
            padded_img[i, j] = a

    return padded_img


def crop_padding(pad_img, kernel_size):
    num_pad = kernel_size // 2
    height = pad_img.shape[0] - 2 * num_pad
    width = pad_img.shape[1] - 2 * num_pad
    cropped_img = pad_img[num_pad: height + num_pad, num_pad: width + num_pad]
    cropped_img = cropped_img.astype(pad_img.dtype)

    return cropped_img

def generate_Gaussian_array(odd_size, sigma):
    half_odd_size = odd_size//2
    gaussian_array = []
    gaussian_normalizer = 0
    for i in range(-half_odd_size, half_odd_size + 1):
        gaussian_value = 1/np.sqrt(2*3.14159)/sigma*np.exp(
                        -1*i**2/2/sigma**2)
        gaussian_array.append(gaussian_value)
        gaussian_normalizer += gaussian_value
    gaussian_array = np.array(gaussian_array) / gaussian_normalizer

    return gaussian_array


def fast_Gaussian_filter(image, kernel, sigma):
    # image is a list of list, kernel is an odd number
    # zero padding the image
    img_pad = img_padding(image, kernel)
    # generate gaussian array
    gaussian_array = generate_Gaussian_array(kernel, sigma)

    hor_range = len(img_pad[0]) - kernel
    ver_range = len(img_pad) - kernel
    # horizontal convolution
    for i in range(kernel//2, len(img_pad)-kernel//2):    # row
        for j in range(0, hor_range):   # column
            h_conv = img_pad[i, j:j+kernel]
            h_value = np.dot(h_conv, gaussian_array)
            img_pad[i, j+kernel//2] = h_value

    # vertical convolution
    for k in range(kernel//2, len(img_pad[0])-kernel//2): # column
        for l in range(0, ver_range):   # row
            v_conv = img_pad[l:l+kernel, k]
            v_value = np.dot(v_conv, gaussian_array)
            img_pad[l+kernel//2, k] = v_value

    # crop the padded image
    blurred_image = crop_padding(img_pad, kernel)

    return blurred_image


def subsample(img2sub):
    subbed_image = []

    for i in range(0, len(img2sub), 2):
        imgRow = []
        for j in range(0, len(img2sub[0]), 2):
            pixel = img2sub[i][j]
            imgRow.append(pixel)

        subbed_image.append(imgRow)
    subbed_col = len(imgRow)
    subbed_image = np.asarray(subbed_image)
    subbed_image = subbed_image.reshape(-1, subbed_col)
    subbed_image = subbed_image.astype(img2sub.dtype)
    return subbed_image


def generate_Gaussian_Pyramid(start_img, kernel=7):
    NUM_OCTAVES = 4
    Pyramid = []
    sigma0 = 1.6
    S = 3
    k = 2**(1/S)

    for oct_num in range(NUM_OCTAVES):
        octave_img = []
        for j in range(5):
            sigma = sigma0 * 2**oct_num * k**(j+1)
            #img_blur = fast_Gaussian_filter(start_img, kernel, sigma)
            img_blur = cv2.GaussianBlur(start_img, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
            octave_img.append(img_blur)

        start_img = subsample(start_img)
        Pyramid.append(octave_img)
    return Pyramid


def diffGaussian(pyramids):
    diff_Gaussian_pyramid = []

    for oct_num in range(len(pyramids)):
        diff_octave = []
        for pic_no in range(len(pyramids[0])-1):
            oct_gauss1 = np.asarray(pyramids[oct_num][pic_no])
            oct_gauss2 = np.asarray(pyramids[oct_num][pic_no+1])
            difference = oct_gauss1 - oct_gauss2
            diff_octave.append(difference)

        diff_Gaussian_pyramid.append(diff_octave)

    return diff_Gaussian_pyramid


# DoG
pyramid = generate_Gaussian_Pyramid(imageArray, 7)
diff_pyramid = diffGaussian(pyramid)

# Display and save DoG
for pyd_oct in range(len(diff_pyramid)):
    for oct_pic in range(len(diff_pyramid[0])):
        cv2.imshow('Dog', diff_pyramid[pyd_oct][oct_pic])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('Results/DoG_Octave'+str(pyd_oct)+'_Pic'+str(oct_pic)+'.jpg',
                    diff_pyramid[pyd_oct][oct_pic])

