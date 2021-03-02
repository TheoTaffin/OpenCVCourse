import cv2
import os

import config


### Loading image

relative_img_path = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and-TensorFlow-2" \
                  "/Section_1/images/input.jpg"
img_path = os.path.join(config.root_path, relative_img_path)
image = cv2.imread(img_path)

### prompting image

# To display an image, we need to give a title to the window and the image
win_name = 'image'
cv2.imshow(win_name, image)

# It is important to use cv2.waitKey(), bcs it allows to input information if we want. And it
# avoid crashes as well. Using 0 is the same as leaving it blank
cv2.waitKey(0)

# don't forget that or the program will hang
cv2.destroyAllWindows()


### Let's check on the image
# (height (rows), width(cols), channel) -> (597, 968, 3)
print(image.shape)

### Let's store it with two different format
relative_output_img_path_jpg = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and" \
                              "-TensorFlow-2/Section_1/images/output.jpg"
relative_output_img_path_png = "github/Hands-On-Computer-Vision-with-OpenCV-4-Keras-and" \
                               "-TensorFlow-2/Section_1/images/output.png"


output_path_jpg = os.path.join(config.root_path, relative_output_img_path_jpg)
output_path_png = os.path.join(config.root_path, relative_output_img_path_png)

cv2.imwrite(output_path_png, image)
cv2.imwrite(output_path_jpg, image)

# note : .png takes way more space than .jpg
