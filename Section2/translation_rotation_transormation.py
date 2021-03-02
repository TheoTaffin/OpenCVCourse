import os
import config
import cv2
import numpy as np

# Translation are geometric distortions enacted upon an image
# Are used to correct distortions or perspective issues from arising from the point of view an
# image was captured
# Two main type :
## Affine : scaling, rotation, translation, skewing (transforming)
## In affine translation, parallelism is conserved
## None Affine (Projective transform, Homography) : does not preserve parallelism, length and angle
# but conserves collinearity and incidence


relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                  "/Section_2/images/input.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)

# we store height and width and calculate a quarter of each to shift it by that
height, width = image.shape[0:2]
quarter_h, quarter_w = height/4, width/4

# Translation matrix
T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])

img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow("Translation", img_translation)
cv2.waitKey()
cv2.destroyAllWindows()


# Rotation
# width and height /2 -> center of the images
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)

# input rotation matrix to warp affine
img_rotation_1 = cv2.warpAffine(image, rotation_matrix, (width, height))
cv2.imshow("Rotation 1", img_rotation_1)
cv2.waitKey()
cv2.destroyAllWindows()


# Cropping

# Transposition of a matrix does basically a 90° flip counter clockwise
rotated_img_2 = cv2.transpose(image)

cv2.imshow('Rotation 2', rotated_img_2)
cv2.waitKey()
cv2.destroyAllWindows()


# Using flip to do an horizontal flip (arg for flipping method)
flipped = cv2.flip(image, -1)
cv2.imshow('Flipped', flipped)
cv2.waitKey()
cv2.destroyAllWindows()
