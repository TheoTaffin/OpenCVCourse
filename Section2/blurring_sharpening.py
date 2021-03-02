import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/oxford.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)


# blurring based on convolution principle
# an operation where we average the pixels within a region (kernel)
# blurring smooths images and can reduce noise, ease edge detection

# cv2.blur : average images over a specified window
# cv2.GaussianBlur : gives more relevance to the closer pixels

# [1, 1, 1]
# [1, 1, 1]  * 1/9
# [1, 1, 1]
# here we normalize to keep the brightness at the same level as the original

# sharpening is the opposite of blurring, it strengthens or emphasizes edges in an image

# [-1, -1, -1]
# [-1, 9, -1]
# [-1, -1, -1]
# here normalization is not needed, cuz the matrix sums to one


# creating a 3x3 kernel
kernel_3x3 = np.ones((3, 3), np.float32) * 1/9

# we use cv2.filter2D to convolve the kernel with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('blurred', blurred)
cv2.waitKey()


# trying with a larger kernel
# creating a 3x3 kernel
kernel_3x3 = np.ones((7, 7), np.float32) * 1/49

# we use cv2.filter2D to convolve the kernel with an image
blurredx7 = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('blurredx7', blurredx7)
cv2.waitKey()

# same for sharpen, we change the matrix
kernel_3x3 = np.ones((3, 3), np.float32) * -1
kernel_3x3[2][2] = 9

# we use cv2.filter2D to convolve the kernel with an image
sharpened = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('sharpened', sharpened)
cv2.waitKey()

# same
kernel_3x3 = np.ones((7, 7), np.float32) * -1
kernel_3x3[4][4] = 49

# we use cv2.filter2D to convolve the kernel with an image
sharpened = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('sharpened', sharpened)
cv2.waitKey()

cv2.destroyAllWindows()

# using cv2 blurring methods to see different results now
blur = cv2.blur(image, (3, 3))
cv2.imshow('blur', blur)
cv2.waitKey()


gaussian = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow('gaussian', gaussian)
cv2.waitKey()

# takes the median of all pixels in the kernel to replace the central value with it
median = cv2.medianBlur(image, 3, 0)
cv2.imshow('median', median)
cv2.waitKey()

# bilateral is very effective in noise removal while keeping edges sharp (lots of details kept,
# not a lot of details smoothed)
bilateral = cv2.bilateralFilter(image, 3, 75, 75)
cv2.imshow('bilateral', bilateral)
cv2.waitKey()

cv2.destroyAllWindows()


# image de-noising

dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)

cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey()

cv2.destroyAllWindows()



