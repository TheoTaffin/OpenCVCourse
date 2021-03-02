import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/input.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)


# the point here is to show the difference between using cv2.add and just addition between image
# matrix
M = np.ones(image.shape, dtype="uint8") * 100
print(M)


added = cv2.add(image, M)
cv2.imshow("w/ cv2.add", added)
cv2.waitKey()

added_2 = image + M
cv2.imshow("w/o cv2.add", added_2)
cv2.waitKey()

cv2.destroyAllWindows()

# ccl : when we use cv2.add, cv2 caps the value to 255. Overwise, imread interprets value higher
# than 255 as going back to 0 and starting again, so the results doesn't rly make any sense.
# Similar behavior with cv2.subsract and image - M.


# Bitwise and masking operations

# Making a sqare
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow("square", square)
cv2.waitKey()

# Making an ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("ellipse", ellipse)
cv2.waitKey()

cv2.destroyAllWindows()


# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", And)
cv2.waitKey()

# Shows where either square or ellipse is
bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey()

# Shows where either exist by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey()

# Shows everything that isn't a part of the squart
bitwiseNot = cv2.bitwise_not(square)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey()

cv2.destroyAllWindows()