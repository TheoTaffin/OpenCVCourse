import os
import config
import cv2
import numpy as np

relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_3/images/bunchofshapes.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)

# Sorting contours is quite useful :
## sorting by area :
# ## - you can eliminate small contours that may be noise
# ## - Extract largest contour
## sorting by spatial position (using the contour centroid)
# ## - sort characters left to right
# ## - process images in specific order

# Contours moments :
# Moments are a set of scalers that are an aggregate of a set of vectors
# Analogie as a measure of image intensities
# image with pixel intensities I(x, y), moments are given by :
# Mij = SUMx(SUMy(I(x, y))

# We can use the image moments to find the centroid of contours by
# Cx = M10 / M00, Cy = M01 / M00
# Cx and Cy are respectively the x-cco and y-coo of the centroid


# Approximating contours is useful when correcting slight distortions in your contour
# We use approxPolyDP to achieve this
# cv2.approxPolyDP(contour, Approximation Accuracy, Closed)

blank_img = np.zeros((image.shape))

cv2.imshow("original", image)
cv2.waitKey()

# preprocessing pipeline for contour
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey()


# Find canny edges
edged = cv2.Canny(gray, 50, 200)
cv2.imshow("canny edge", edged)
cv2.waitKey()

# find contour
edged_copy = edged.copy()
contours, hierarchy = cv2.findContours(edged_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(f"number of contours found : {len(contours)}")


# draw over blankk
cv2.drawContours(blank_img, contours, -1, (0, 255, 0), thickness=3)
cv2.imshow("contours over blank", blank_img)
cv2.waitKey()

# Draw over original
original_copy = image.copy()
cv2.drawContours(original_copy, contours, -1, (0, 255, 0), thickness=3)
cv2.imshow("contours over original", original_copy)
cv2.waitKey()

cv2.imshow("original", image)
cv2.waitKey(0)


all_areas = [cv2.contourArea(cnt) for cnt in contours]

print("contour areas before sorting")
print(all_areas)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

all_areas_sorted = [cv2.contourArea(cnt) for cnt in sorted_contours]

print("contour areas before sorting")
print(all_areas_sorted)

original_copy = image.copy()
for count, cnt in enumerate(sorted_contours):
    cv2.drawContours(original_copy, [cnt], -1, (255, 0, 0), thickness=3)
    cv2.imshow(f"contour {count}", original_copy)
    cv2.waitKey()


cv2.destroyAllWindows()


# Sorting contours left ro right (using centroid)
contour_centroid_x_y = [(
    cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'],
    cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00']
)
    for cnt in contours]

original_copy = image.copy()
for x, y in contour_centroid_x_y:
    cx = int(x)
    cy = int(y)
    cv2.circle(original_copy, (cx, cy), 10, (0, 255, 0), 3)
    cv2.imshow(f"{x, y}", original_copy)
    cv2.waitKey()


contours_left_to_right = sorted(contours,
                                key=lambda x_c: cv2.moments(x_c)['m01']/cv2.moments(x_c)['m00'],
                                reverse=False)

original_copy = image.copy()
for count, cnt in enumerate(contours_left_to_right):
    cv2.drawContours(original_copy, [cnt], -1, (255, 0, 0), thickness=3)
    cx = int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])
    cy = int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])

    cv2.putText(original_copy, str(count+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('sorting l to r', original_copy)
    cv2.waitKey()
    (x, y, w, h) = cv2.boundingRect(cnt)

cv2.destroyAllWindows()


# approximate PolyDP
relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_3/images/house.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("gray", gray)
cv2.waitKey()

cv2.imshow("thresh", thresh)
cv2.waitKey()


# find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
copy = image.copy()


for cnt in contours:
    cv2.drawContours(copy, [cnt], -1, (0, 255, 0), 3)
    cv2.imshow('bounding', copy)

cv2.waitKey()

copy = image.copy()
for cnt in contours:
    accuracy = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy, True)
    cv2.drawContours(copy, [approx], 0, (0, 255, 0), 3)
    cv2.imshow('approx', copy)

cv2.waitKey()

cv2.destroyAllWindows()


# Convex HUll
# looks similar to contour approximation, but it is not. Here, cv2.convexHull() function checks
# a curve for convexity defects and corrects it. Generally speaking, convex curves are the
# curves which are always bulged out, or at least flat. And if it is bulged inside, it is called
# convexity defects. For example, in the below image of hand. Red line shows the convex hull of
# hand. The double sided arrow marks shows the convexity defects, chich are the local maximum
# deviations of hull from contours


# approximate PolyDP
relative_image_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                      "/Section_3/images/hand.jpg"
root_image_path = os.path.join(config.root_path, relative_image_path)
image = cv2.imread(root_image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 176, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("gray", gray)
cv2.waitKey()

cv2.imshow("thresh", thresh)
cv2.waitKey()


# find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
copy = image.copy()


n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

cpy = image.copy()
cv2.drawContours(cpy, contours, -1, (0, 255, 0), 2)
cv2.imshow("cnt classic", cpy)

for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow('Convex Hull', image)
    cv2.waitKey()


cv2.destroyAllWindows()