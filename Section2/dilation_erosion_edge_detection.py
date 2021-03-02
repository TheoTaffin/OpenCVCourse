import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_2/images/gradient.jpg"
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


