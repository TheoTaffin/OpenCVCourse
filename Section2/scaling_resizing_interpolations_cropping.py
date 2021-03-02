import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/oxford.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)

# note
# cv2.resize(image, dsize(output_size), x_scale, y_scale, interpolation
# Interpolation methods :
# cv2.INTER_AREA - (good for shrinking or down sampling
# cv2.INTER_NEAREST - Fastest
# cv2.INTER_LINEAR - Good for zooming or up sampling (default)
# cv2.INTER_CUBIC - Better
# cv2.INTER_LANCZOS4 - Best


### part 1
# resizing at 3/4 of the original size
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
cv2.imshow("3/4 interpolation default (INTER_LINEAR)", image_scaled)
cv2.waitKey()


# Doubling the size
image_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("3/4 interpolation default (INTER_CUBIC)", image_scaled)
cv2.waitKey()


# Doubling the size
image_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imshow("3/4 interpolation default (INTER_NEAREST)", image_scaled)
cv2.waitKey()


# Skkewing by setting exact dimension
image_scaled = cv2.resize(image, (900, 400), interpolation=cv2.INTER_NEAREST)
cv2.imshow("3/4 interpolation default (INTER_NEAREST)", image_scaled)
cv2.waitKey()

cv2.destroyAllWindows()


# Image Pyramids (default 1/2 and 2 respectively)
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

cv2.imshow("original", image)
cv2.imshow("pyrdown", smaller)
cv2.imshow("ppydup", larger)
cv2.waitKey()

cv2.destroyAllWindows()


# Cropping
height, width = image.shape[0:2]

# getting start and end coordinates (top left bottom right)
start_row, start_col = int(height * 0.25), int(width * 0.25)
end_row, end_col = int(height * 0.75), int(width * 0.75)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("original", image)
cv2.waitKey()

# we store the image bcs cv2.rectangle draws a rectangle directly in the original image (
# in-place operation)
copy = image.copy()
cv2.rectangle(copy, (start_col, start_row), (end_col, end_row), (255, 0, 0), 3)
cv2.imshow("cropped area", copy)
cv2.waitKey()

cv2.imshow("cropped image", cropped)
cv2.waitKey()

cv2.destroyAllWindows()

