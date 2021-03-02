import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/gradient.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)

# Binarization : changing intensity value to either 0 or 1 (0 - 255)

# image needs to be converted to grayscale b4 thresholding
# Thresholding is the act of converting an image to a binary form
# same as binarization but we set the value ourselves for the threshold and the threshold type
# cv2.THRESH_BINARY : same as binarization
# cv2.THRESH_BINARY_INV : inverse
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV

# biggest downfal from thresholding is that we provide threshold values. Adaptive thresholding
# take that uncertainty away if we don't know what the threshold should be.
# cv2.adaptiveThreshold

# types:
# ADAPTIVE_THRESH_MEAN_C : takes mean based on the neighborhood pixels
# ADAPTIVE_THRESH_GAUSSIAN_C : weigthed sum of neighborhood pixels under gaussian window
# THRESH_OTSU : (uses cv2.threshold function)assumes there are two peaks in the grayscale histogram
# of the image, and then
# tries to find an optimal value to separate these to peaks to find T

cv2.imshow('original', image)
cv2.waitKey(0)


# values below 127 goes to 0 (black, rest white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1 threshold binary', thresh1)
cv2.waitKey(0)

# values below 127 goes to 1 (white, rest black)
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('1 threshold binary inv', thresh2)
cv2.waitKey(0)


# values above 127 are truncated (kept) at 127 (capping max brightness values)
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('1 threshold truncated', thresh3)
cv2.waitKey(0)

# values below 127 goes to 0, above 127 are unchanged
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('1 threshold tozero', thresh4)
cv2.waitKey(0)

# values above 127 goes to 0, above 127 are unchanged
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('1 threshold tozero inv', thresh5)
cv2.waitKey(0)


cv2.destroyAllWindows()


relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/Origin_of_Species.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)

# adaptive thresholding
cv2.imshow('original', image)
cv2.waitKey(0)

# good practive to blur the image as it removes the noise
image = cv2.GaussianBlur(image, (3, 3), 0)

# using adaptiveTreshold
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
cv2.imshow("ADAPTIVE MEAN THRESHOLDING", thresh)
cv2.waitKey()

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("ADAPTIVE THRESH_OTSU", th2)
cv2.waitKey()

# otsu's thresholding after gaussian filtering
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("ADAPTIVE THRESH_OTSU 2", th3)
cv2.waitKey()

cv2.destroyAllWindows()