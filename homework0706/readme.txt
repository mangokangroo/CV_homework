median_blur.py is a Python script to achieve median blur function of a specified colorful picture.

The function defined in the script "medianBlur(img, kernel, padding_way)" is composed of two parts:
	a. ZEROS padding and REPLICA padding of the matrix of the image
	b. find median values consecutively of the kernel-size part of the padded matrix, and assign them to the central pixel

Program inputs include:
1. the location of the targeted image
2. size of the kernel(sliding window) for median blurring(like 5 3, both should be odd integers, seperator is " ¡±, and the first number 
						should be no less than the second)
3. the approach for padding, two options are available "ZEROS" "REPLICA"

Outputs:
1. the original picture
2. the median blurred picture