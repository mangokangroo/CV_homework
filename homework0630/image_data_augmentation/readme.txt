image_processing.py is a Python script to achieve data augmentation of a specified picture.

The functions of the script include: a. image crop
			      b. color shift
			      c. rotation
			      d. perspective transform

Inputs include(reminders are promted where needed): 
1. the location of the targeted image
2. the shifting coefficient of the basic components of color, red, green, and blue, respectively( must be INTEGER)
3. the angle for rotation(in degree, positive value means counterclockwise rotation)
4. the margin value for the perspective transform

Outputs include:
1. image after cropping
2. image after color shifting (based on output 1)
3. image after rotation (based on output 2)
4. image after perspective transformation (based on output 3)

Notes:
1. the default range for image cropping is [0:150, 0:200], which can only be changed in the code by far