import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/opencv_inv.png"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)


# dilation adds pixels to the boundary of objects in an image
# erosion removes pixels to the boundary of objects in an image

# opening erosion followed by dilation
# closing dilation followed by erosion

# removing pixels means getting them to black, which means that dilation on a black text for
# instance will actually thickens the text itself


# edge detection : edges are sudden changes (discontinuities) in an image and they can encode
# just as much information as pixels
# few edges detection algo :
# Sobel - to emphasize vertical or horizontal edges
# Laplacian - gets all
# Canny - Optimal due to low error rate, well defined edges and accurate detection

# Canny :
# apply gaussian blurring
# finds intensity gradient of the image
# applied non-maximum suppression (that is, removes pixels that are not edges)
# Hysteresis - Applies thresholds (that is, if pixel is withing the upper or lower thresholds,
# it is considered an edges)

cv2.imshow('original', image)
cv2.waitKey()


kernel = np.ones((9, 9), np.uint8)

# erosion
erosion = cv2.erode(image, kernel, iterations=1)
cv2.imshow('erosion', erosion)
cv2.waitKey()


# dilation
dilation = cv2.dilate(image, kernel, iterations=1)
cv2.imshow("dilation", dilation)
cv2.waitKey()


# Opening - Good for removing noise (erosion then dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey()


# Closing - Good for removing noise (dilation then erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)
cv2.waitKey()

cv2.destroyAllWindows()



# Edge detecion
relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/oxford.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)


# Extract Sobel Edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow('Original', image)
cv2.waitKey()
cv2.imshow('Sobel X', sobel_x)
cv2.waitKey()
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey()


sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_OR', sobel_OR)
cv2.waitKey()


laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('laplacian', laplacian)
cv2.waitKey()


# For canny, we need to provide two values, thresh 1 and thresh 2. Any gradient value larger
# than thresh2 is considered to be an edge. Any value below thresh 1 is considered not to be an
# edge. Values in between are either classified as edfes or non edges based on how their
# intensities are "connected. in this case, any gradient values below 60 are considered
# non-edges whereas any values above 120 are considered edges

canny = cv2.Canny(image, 50, 120)
cv2.imshow('canny', canny)
cv2.waitKey()


canny_large = cv2.Canny(image, 10, 200)
cv2.imshow('canny_large', canny_large)
cv2.waitKey()


canny_narrow = cv2.Canny(image, 100, 120)
cv2.imshow('canny_narrow', canny_narrow)
cv2.waitKey()


cv2.destroyAllWindows()
