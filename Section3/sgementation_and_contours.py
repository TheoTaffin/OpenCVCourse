import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_3/images/bunchofshapes.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path, 0)


# Segmentation is the process by which we partition or segment sections of an image into regions
# Contours are continuous lines or curves that bound or cover the full boundary of an object in
# an image, having same color or intensity.
# contour algo :
# - in opencv, the findContours algo seeks to find or locate regions of contrasting intensity,
# for example, finding a white object from a black background:
## - Generally the to be found should be white and background should be black
## - For better acc, use binary images, so apply threshold for canny edges detections for instance
# opencv returns a list of lists (array of arrays) of contours

# cv2.findContours (image, retrieval Mode, Approximation Method)
# return - contours, hierarchy :
## - Contours are stored as a numpy array of (x, y) points that form the contour
## - Hierarchy describes the child-parent relationships between contours (that is, contours
## within contours)

# Approximation Mode :
## - cv2.CHAIN_APPROX_NONE - store all the points along the line (inefficient)
## - cv2.CHAIN_APPROX_SIMPLE - Stores the end points of each line

# Hierarchy types (first two are the most useful):
## - cv2.RETR_LIST - retrieves all contours
## - cv2.RETR_EXTERNAL - retrieves external or outer contours only
## - cv2.RETR_COMP - retrieves all in a 2-level hierarchy
## - cv2.RETR_TREE - all in full hierarchy

# Hierarchy : [next, previous, first child, parent]

# cv2.drawContours(image, contours, specific contours, color, thickness)

cv2.imshow("original", image)
cv2.waitKey()


# preprocessing pipeline for contour
# Grayscale
gray = image
cv2.imshow("gray", gray)
cv2.waitKey()


# Find canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("canny edge", edged)
cv2.waitKey()


# Finding Contours
# We use a copy of our image (edged.copy()), so we don't affect the input
# reminder : retr_external -> find outter contours only, chain_approx_none -> retrieves all
# points of the contours
tmp_copy = edged.copy()
contours, hierarchy = cv2.findContours(tmp_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow("canny edge after contouring", tmp_copy)
cv2.waitKey()

print(f"number of contours found = {len(contours)}")


# Draw all contours
# Use "-1" as the 3rd parameter to raw
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
cv2.imshow("contours", image)
cv2.waitKey()

cv2.destroyAllWindows()


