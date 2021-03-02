import cv2
import os
import numpy as np

import config


### Loading image

relative_img_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                    "/Section_1/images/input.jpg"
img_path = os.path.join(config.root_path, relative_img_path)
image = cv2.imread(img_path)

# RGB : Red Green Blue (BGR in openCV)
# additive color model :
## Red (0 - 255)
## Green (0 - 255)
## Yellow (0 - 255)

# HSV : Hue, Saturation, Value(Brightness)
# attempts to represent colors the way humans perceive it
## hue - Color value (0 - 179)
## Saturation - vibrancy of color (0 - 255)
## Value - Brightness (0 - 255)

## Color range filter (Hue) :
### Red 165 - 15
### Green 45 - 75
### Blue 90 - 120

# Gray scaling : black and white image
# Represented by a 2-D array
## 0 : Black
## 255 : White


cv2.imshow('Original', image)
cv2.waitKey()

# cvtColor, takes image and converter in args
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray scaled', gray_image)
cv2.waitKey()

cv2.destroyAllWindows()


# Split color components
B, G, R = cv2.split(image)
bgr_lst = ["B", "G", "R"]
# show each filter in grayscale:
for color, img_filter in zip(bgr_lst, cv2.split(image)):
    cv2.imshow(color, img_filter)
    cv2.waitKey()


def show_img(title, blue, red, green):
    cv2.imshow(title, cv2.merge([blue, red, green]))
    cv2.waitKey()


# creating blank image with a matrix of zeros
zeros = np.zeros(image.shape[0:2], dtype="uint8")

show_img("Red", zeros, zeros, R)
show_img("Green", zeros, G, zeros)
show_img("Blue", B, zeros, zeros)

# original
show_img("Red, Blue", B, G, R)

# inc blue
show_img("Blue +", B+100, G, R)

# inc red
show_img("Red +", B, G, R+100)

# inc green
show_img("Green +", B, G+100, R)

cv2.waitKey()
cv2.destroyAllWindows()


# HSV

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# view each channel of the HSV converted image
cv2.imshow('HSV image', hsv_image)
cv2.imshow('Hue Channel', hsv_image[:, :, 0])
cv2.imshow('Saturation Channel', hsv_image[:, :, 1])
cv2.imshow('Value Channel', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()