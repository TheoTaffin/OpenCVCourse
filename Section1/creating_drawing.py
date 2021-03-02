import cv2
import numpy as np

# Create a black image using numpy
image = np.zeros((512, 512, 3), np.uint8)

# Can we make this in black and white ?
image_bw = np.zeros((512, 512), np.uint8)


# Same, just on of those image is using 3 channels
cv2.imshow("color", image)
cv2.imshow("b&w", image_bw)

cv2.waitKey()
cv2.destroyAllWindows()


# Drawing a diagonal blue line of thickness of 5 pixels
image = np.zeros((512, 512, 3), np.uint8)

# Image input, start and end point, color, line thckness
cv2.line(image, (0, 0), (511, 511), (255, 127, 0), 5)
cv2.imshow("Blue Line", image)
cv2.waitKey()
cv2.destroyAllWindows()


# Draw a rectangle
image = np.zeros((512, 512, 3), np.uint8)

# image, higher left, lower right, color, thickness (-1 is filled)
cv2.rectangle(image, (100, 100), (350, 250), (255, 127, 0), 10)
cv2.imshow("Blue rectangle", image)
cv2.waitKey()
cv2.destroyAllWindows()


# Draw a circle
image = np.zeros((512, 512, 3), np.uint8)

# image, center, radius, color, thickness (-1 is filled)
cv2.circle(image, (255, 255), 100, (255, 127, 0), 10)
cv2.imshow("Blue circle", image)
cv2.waitKey()
cv2.destroyAllWindows()


# Draw a polygon
image = np.zeros((512, 512, 3), np.uint8)

# Let's define four points
pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)

# reshaping for poly lines requirements, consistent with multiple cv2 format requirements
pts = pts.reshape((-1, 1, 2))

# image, points, bool (connected or not), color, thickness, (special function for filled ones)
cv2.polylines(image, [pts], True, (255, 127, 0), 10)
cv2.imshow("Blue polygon", image)
cv2.waitKey()
cv2.destroyAllWindows()


### Add some text
# Draw a polygon
image = np.zeros((512, 512, 3), np.uint8)

# image, string, starting point, font, font size, color, thickness
cv2.putText(image, 'Hello World', (75, 290), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
cv2.imshow("Hello World", image)
cv2.waitKey()
cv2.destroyAllWindows()
